"""Q5 isolated, resumable dual-GPU experiments and evidence helpers."""
from __future__ import annotations
import argparse, gc, hashlib, importlib, importlib.util, json, math, os, random
import shutil, subprocess, sys, tempfile, time, uuid, zipfile
from array import array
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from common import (BPETokenizerWrapper, TokenDataset, held_out_sentences,
                    load_corpus, perplexity_of_texts, remove_sentences_from_text)
from model_builtin import BuiltinDecoderLM, BuiltinLMConfig, sinusoidal_positional_encoding

STAGES = ('bootstrap','dependency_validation','builtin_model_generation','unit_tests',
          'attention_equivalence','numerical_debugging','smoke_tests',
          'builtin_training','scratch_baseline_validation','matched_comparison',
          'architecture_comparison','limitation_analysis','retrofit_analysis',
          'timing_benchmark','plots','submission','autograder','recovery_export','complete')
SCHEMA = 1

def now(): return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''): h.update(part)
    return h.hexdigest()

def jhash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def hash_ints(values):
    h=hashlib.sha256()
    for start in range(0,len(values),100000):
        h.update(array('q',(int(x) for x in values[start:start+100000])).tobytes())
    return h.hexdigest()

def hash_texts(texts):
    h=hashlib.sha256()
    for text in texts:
        b=str(text).encode();h.update(len(b).to_bytes(8,'little'));h.update(b)
    return h.hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
        json.loads(tmp.read_text(encoding='utf-8'));os.replace(tmp,path)
    finally: tmp.unlink(missing_ok=True)

def load_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def torch_load(path):
    try:return torch.load(path,map_location='cpu',weights_only=False)
    except TypeError:return torch.load(path,map_location='cpu')

def write_torch(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        torch.save(value,tmp);torch_load(tmp);os.replace(tmp,path)
    finally:tmp.unlink(missing_ok=True)

def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value

def scratch_stack(root):
    source=Path(root)/'artifacts/full/q5/source_snapshots/q1_scratch'
    if str(source) not in sys.path:sys.path.insert(0,str(source))
    for name in ('attention','model_arch'):
        mod=sys.modules.get(name)
        if mod is not None and Path(getattr(mod,'__file__','')).resolve()!=source/(name+'.py'):
            sys.modules.pop(name,None)
    return importlib.import_module('model_arch'),importlib.import_module('attention')

def tokenizer_from_snapshot(root):
    source=Path(root)/'artifacts/full/q5/source_snapshots/q1_scratch/a1_tokenizer.py'
    spec=importlib.util.spec_from_file_location('q5_a1_tokenizer',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    impl=module.Assignment1ByteLevelBPE.load(
        Path(root)/'artifacts/shared/tokenizer/a1_bpe_tokenizer.json')
    return BPETokenizerWrapper(impl)

def model_config(q1_arch):
    fields=('vocab_size','d_model','n_layers','n_heads','d_ff','max_seq_len','dropout','norm_first')
    return {k:q1_arch[k] for k in fields}

def scratch_config(q1_arch,mod):return mod.LMConfig(**q1_arch)

def mapped_builtin(scratch,builtin):
    with torch.no_grad():
        builtin.tok_emb.weight.copy_(scratch.tok_emb.weight)
        builtin.ln_f.load_state_dict(scratch.ln_f.state_dict())
        for src,dst in zip(scratch.blocks,builtin.encoder.layers):
            dst.self_attn.in_proj_weight.copy_(torch.cat([
                src.attn.q_proj.weight,src.attn.k_proj.weight,src.attn.v_proj.weight],dim=0))
            dst.self_attn.out_proj.weight.copy_(src.attn.out_proj.weight)
            dst.self_attn.in_proj_bias.zero_();dst.self_attn.out_proj.bias.zero_()
            dst.linear1.load_state_dict(src.ffn.net[0].state_dict())
            dst.linear2.load_state_dict(src.ffn.net[3].state_dict())
            dst.norm1.load_state_dict(src.ln1.state_dict())
            dst.norm2.load_state_dict(src.ln2.state_dict())
    return builtin

def count_trainable(model):return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Q5Runtime:
    def __init__(self,root,archive_refs):
        self.root=Path(root).resolve();self.base=self.root/'artifacts/full/q5'
        self.base.mkdir(parents=True,exist_ok=True)
        self.smoke=self.root/'artifacts/smoke/q5';self.smoke.mkdir(parents=True,exist_ok=True)
        self.status_path=self.base/'pipeline_status.json'
        self.refs=archive_refs
        if self.status_path.is_file():self.status=load_json(self.status_path)
        else:self.status={'schema':SCHEMA,'stages':{s:{'state':'pending'} for s in STAGES}}
        self.stage('bootstrap','completed',root=str(self.root),archives=self.refs)
        self.q1=load_json(self.root/'artifacts/full/q1/standard/config.json')
        self.q1_metrics=load_json(self.root/'artifacts/full/q1/standard/metrics.json')
        arch=self.q1['architecture']
        if not (arch['n_heads']==arch['n_kv_heads']==4 and arch['pos_emb_type']=='sinusoidal'
                and arch['attention_scale'] is None and self.q1['target_steps']==500
                and self.q1['batch_size']==32 and self.q1['optimizer']=={'name':'Adam','lr':0.0003}):
            raise ValueError('Q1 baseline differs from verified standard-MHA/sinusoidal contract')
        self.arch=arch;self.cfg=model_config(arch)
        for name in ('attention.py','model_arch.py','a1_tokenizer.py'):
            expected=self.q1['source_hashes'][name]
            actual=sha(self.base/'source_snapshots/q1_scratch'/name)
            if actual!=expected:
                raise ValueError(f'Q1 source fingerprint mismatch for {name}')
        if (self.q1_metrics.get('completed_steps')!=500 or
            not math.isfinite(float(self.q1_metrics.get('held_out_ppl',float('nan'))))):
            raise ValueError('Q1 baseline metrics are incomplete')
        self.spec={'schema':SCHEMA,'architecture':self.cfg,'q1_architecture':arch,
                   'steps':self.q1['target_steps'],'batch_size':self.q1['batch_size'],
                   'optimizer':self.q1['optimizer'],'seed':self.q1['seed'],
                   'sampler_scheme':self.q1['sampler_scheme'],
                   'tokenizer_fingerprint':self.q1['tokenizer_fingerprint'],
                   'train_ids_hash':self.q1['train_ids_hash'],
                   'held_out_hash':self.q1['held_out_hash'],
                   'source_hashes':{n:sha(self.base/'source_snapshots/q1_scratch'/n)
                                    for n in ('attention.py','model_arch.py','a1_tokenizer.py')},
                   'builtin_source_hash':sha(self.root/'model_builtin.py'),
                   'runtime_source_hash':sha(self.root/'q5_runtime.py'),
                   'precision':'cuda_fp16_amp_else_fp32',
                   'torch_version':torch.__version__}
        self.spec['fingerprint']=jhash(self.spec)
        spec_path=self.base/'experiment_spec.json'
        if spec_path.is_file() and load_json(spec_path)['fingerprint']!=self.spec['fingerprint']:
            raise RuntimeError('Q5 experiment fingerprint changed; existing runs cannot be mixed')
        write_json(spec_path,self.spec)
        self.stage('builtin_model_generation','completed',sha256=self.spec['builtin_source_hash'])

    def stage(self,name,state,**details):
        assert name in STAGES and state in ('pending','running','completed','failed')
        self.status['stages'][name]={'state':state,'updated_utc':now(),**details}
        write_json(self.status_path,self.status)

    def prepare_data(self):
        self.stage('dependency_validation','running')
        raw=self.root/'data/raw'
        if len(list(raw.glob('book*.txt')))!=7:
            subprocess.run([sys.executable,str(self.root/'download_data.py'),'--raw-dir',str(raw)],check=True)
        corpus=load_corpus(str(raw))
        held=held_out_sentences(corpus,n_chapters=5,sentences_per_chapter=6)
        train_text=remove_sentences_from_text(corpus.raw_text,held)
        data_meta=load_json(self.root/'artifacts/shared/tokenizer/train_ids_metadata.json')
        if hashlib.sha256(train_text.encode()).hexdigest()!=data_meta['training_corpus_sha256']:
            raise ValueError('Downloaded corpus differs from Q1 tokenizer training text')
        ids_tensor=torch_load(self.root/'artifacts/shared/tokenizer/train_ids.pt')
        ids=[int(x) for x in ids_tensor.tolist()]
        if hash_ints(ids)!=self.q1['train_ids_hash'] or hash_texts(held)!=self.q1['held_out_hash']:
            raise ValueError('Q1 train/held-out split fingerprint mismatch')
        tok=tokenizer_from_snapshot(self.root)
        meta=load_json(self.root/'artifacts/shared/tokenizer/tokenizer_metadata.json')
        if (meta['model_sha256']!=self.q1['tokenizer_fingerprint'] or
            tok.vocab_size!=self.cfg['vocab_size']):
            raise ValueError('Q1 tokenizer fingerprint or vocabulary mismatch')
        write_json(self.base/'data_manifest.json',{
            'held_out':held,'held_out_hash':hash_texts(held),
            'train_ids_hash':hash_ints(ids),'tokenizer_fingerprint':meta['model_sha256'],
            'raw_files':{p.name:sha(p) for p in sorted(raw.glob('book*.txt'))},
        })
        self.stage('dependency_validation','completed',n_train_ids=len(ids),n_held_out=len(held))
        return ids,held,tok

    def unit_tests(self):
        self.stage('unit_tests','running')
        checks={}
        def check(name,condition):
            checks[name]=bool(condition)
            if not checks[name]:raise AssertionError(f'Q5 unit test failed: {name}')
        cfg=BuiltinLMConfig(vocab_size=31,d_model=16,n_layers=2,n_heads=4,
                            d_ff=32,max_seq_len=12,dropout=0,norm_first=False)
        pe=sinusoidal_positional_encoding(12,16)
        check('positional_shape',tuple(pe.shape)==(12,16))
        check('positional_formula',abs(pe[0,0].item())<1e-7 and abs(pe[0,1].item()-1)<1e-7
              and abs(pe[1,0].item()-math.sin(1))<1e-6)
        for device in ([torch.device('cpu')]+[torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]):
            model=BuiltinDecoderLM(cfg).to(device).eval()
            for length in (1,7,12):
                x=torch.randint(0,31,(2,length),device=device)
                y=model(x)
                check(f'shape_{device}_{length}',tuple(y.shape)==(2,length,31))
                check(f'finite_{device}_{length}',torch.isfinite(y).all().item())
            x=torch.randint(0,31,(2,9),device=device)
            altered=x.clone();altered[:,5:]=torch.randint(0,31,(2,4),device=device)
            with torch.no_grad():
                a,b=model(x),model(altered)
            check(f'causal_future_{device}',torch.allclose(a[:,:5],b[:,:5],atol=1e-5,rtol=1e-5))
            loss=F.cross_entropy(model(x).reshape(-1,31),x.reshape(-1));loss.backward()
            check(f'backward_{device}',math.isfinite(float(loss.item())))
            check(f'gradient_{device}',all(p.grad is not None and torch.isfinite(p.grad).all()
                  for p in model.parameters() if p.requires_grad))
            clone=BuiltinDecoderLM(cfg).to(device);clone.load_state_dict(model.state_dict())
            clone.eval();check(f'state_roundtrip_{device}',torch.allclose(model(x),clone(x)))
            try:model(torch.ones(1,13,dtype=torch.long,device=device))
            except ValueError:check(f'max_length_{device}',True)
            else:check(f'max_length_{device}',False)
            del model,clone;gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
        write_json(self.base/'analysis/unit_tests.json',checks)
        self.stage('unit_tests','completed',checks=len(checks))
        return checks

    def equivalence(self):
        self.stage('attention_equivalence','running')
        _,attention=scratch_stack(self.root)
        generator=torch.Generator().manual_seed(787)
        scratch=attention.CausalSelfAttention(64,8,16,dropout=0,attention_scale=None).eval()
        builtin=torch.nn.MultiheadAttention(64,8,dropout=0,bias=False,batch_first=True).eval()
        with torch.no_grad():
            for p in scratch.parameters():p.copy_(torch.randn(p.shape,generator=generator)*0.02)
            builtin.in_proj_weight.copy_(torch.cat([scratch.q_proj.weight,
                scratch.k_proj.weight,scratch.v_proj.weight],dim=0))
            builtin.out_proj.weight.copy_(scratch.out_proj.weight)
            rows=[]
            for length in (1,7,16):
                x=torch.randn(2,length,64,generator=generator)
                allowed=scratch.causal_mask[:length,:length]
                built,_=builtin(x,x,x,attn_mask=~allowed,need_weights=False)
                raw=scratch(x)
                difference=(built-raw).abs()
                rows.append({'length':length,'max_abs_diff':difference.max().item(),
                             'mean_abs_diff':difference.mean().item()})
        worst=max(row['max_abs_diff'] for row in rows)
        report={'rows':rows,'correctness_max_abs_diff':worst,'target_tolerance':1e-3,
                'passed':worst<1e-3,'projection_order':'Q,K,V',
                'scratch_mask':'True allowed','builtin_mask':'True blocked',
                'dropout':0,'bias':False,'precision':'float32_cpu'}
        write_json(self.base/'analysis/attention_equivalence.json',report)
        if not report['passed']:
            self.stage('attention_equivalence','failed',report=report)
            raise AssertionError(f'Attention mismatch {worst:.3e}; inspect Q/K/V, mask, scaling')
        self.stage('attention_equivalence','completed',max_abs_diff=worst)
        return report

    def run_worker(self,kind,phase,device):
        cmd=[sys.executable,str(self.root/'q5_runtime.py'),'--worker',kind,
             '--phase',phase,'--device',device,'--root',str(self.root)]
        return subprocess.Popen(cmd,cwd=self.root,stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,text=True,bufsize=1)

    def run_pair(self,phase):
        devices=[f'cuda:{i}' for i in range(torch.cuda.device_count())]
        assignments={'scratch':devices[0] if devices else 'cpu',
                     'builtin':devices[1] if len(devices)>1 else (devices[0] if devices else 'cpu')}
        if len(devices)>1:
            jobs={kind:self.run_worker(kind,phase,device) for kind,device in assignments.items()}
            for kind,proc in jobs.items():
                for line in proc.stdout:print(f'[{phase}/{kind}] {line}',end='')
            failures={kind:proc.wait() for kind,proc in jobs.items() if proc.wait()!=0}
            if failures:raise RuntimeError(f'{phase} worker failure: {failures}')
        else:
            for kind,device in assignments.items():
                proc=self.run_worker(kind,phase,device)
                for line in proc.stdout:print(f'[{phase}/{kind}] {line}',end='')
                if proc.wait():raise RuntimeError(f'{phase}/{kind} worker failed')
        return assignments

    def smoke_tests(self):
        self.stage('smoke_tests','running')
        assignments=self.run_pair('smoke')
        for kind in ('scratch','builtin'):
            report=load_json(self.smoke/kind/'metrics.json')
            if report['completed_steps']!=2 or not math.isfinite(report['held_out_ppl']):
                raise AssertionError(f'Invalid smoke {kind} metrics')
        write_json(self.smoke/'report.json',{'passed':True,'assignments':assignments,'utc':now()})
        self.stage('smoke_tests','completed',assignments=assignments)
        self.archive(partial=True)  # portable recovery milestone before expensive work
        return assignments

    def full_training(self):
        self.stage('builtin_training','running')
        assignments=self.run_pair('full')
        rows={k:load_json(self.base/k/'metrics.json') for k in ('scratch','builtin')}
        for kind,row in rows.items():
            if (row['completed_steps']!=self.spec['steps'] or row['tokens_processed']!=
                    self.spec['steps']*self.spec['batch_size']*self.cfg['max_seq_len']):
                raise AssertionError(f'Incomplete {kind} full training')
        self.stage('builtin_training','completed',assignments=assignments)
        self.stage('scratch_baseline_validation','completed',
                   historical_q1_ppl=self.q1_metrics['held_out_ppl'],
                   q5_scratch_ppl=rows['scratch']['held_out_ppl'])
        self.archive(partial=True)  # both finished model/optimizer states are recoverable
        return rows

    def matched_comparison(self):
        self.stage('matched_comparison','running')
        rows={k:load_json(self.base/k/'metrics.json') for k in ('scratch','builtin')}
        result={'q5_runs':rows,'historical_q1':{
            'held_out_ppl':self.q1_metrics['held_out_ppl'],
            'parameter_count':self.q1_metrics['parameter_count'],
            'training_seconds':self.q1_metrics['cumulative_active_seconds'],
            'same_hardware_timing':False},
            'matched':{'architecture':self.cfg,'steps':self.spec['steps'],
                       'tokens':self.spec['steps']*self.spec['batch_size']*self.cfg['max_seq_len'],
                       'optimizer':self.spec['optimizer'],'seed':self.spec['seed'],
                       'precision':self.spec['precision'],
                       'data':self.spec['train_ids_hash'],
                       'held_out':self.spec['held_out_hash']},
            'implementation_difference':'Built-in MHA bias tensors are zero and frozen; '
                'FFN and LayerNorm biases remain trainable. Both models start from mapped '
                'Q1 scratch initialization; numerical kernels/dropout ordering can differ.'}
        write_json(self.base/'analysis/matched_comparison.json',result)
        self.stage('matched_comparison','completed')
        return result

    def architecture_comparison(self):
        self.stage('architecture_comparison','running')
        q2_path=self.root/'artifacts/full/q2/analysis/per_config_results.json'
        q3_path=self.root/'artifacts/full/q3/analysis/q3_ppl_comparison.json'
        if not(q2_path.is_file() and q3_path.is_file()):
            raise FileNotFoundError('Attach q2_all_artifacts.zip and q3_all_artifacts.zip for Q5.4 B')
        q2=load_json(q2_path)['per_config_results']
        q3=load_json(q3_path)
        rows=[x for x in q2 if x['attention_type'] in ('mha','gqa','mqa')]
        if len(rows)!=3 or any(not math.isfinite(x['held_out_ppl']) for x in rows):
            raise ValueError('Incomplete Q2 MHA/GQA/MQA evidence')
        result={'q2':rows,'q3':{'sinusoidal_ppl':q3['sinusoidal'],
                               'rope_ppl':q3['rope']},
                'status':'historical scratch-only architecture-side evidence',
                'direct_builtin_variants':['standard MHA + sinusoidal'],
                'unsupported_high_level':['GQA','MQA','RoPE'],
                'workaround':'Retain exact Q2/Q3 scratch implementations as historical '
                             'evidence; no built-in GQA/MQA/RoPE result is claimed.',
                'not_jointly_matched_with_q5':'Q2/Q3 training context differs from Q1/Q5.'}
        write_json(self.base/'analysis/architecture_comparison.json',result)
        self.stage('architecture_comparison','completed')
        return result

    def timing(self):
        self.stage('timing_benchmark','running')
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
        hardware=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        prior=self.base/'analysis/timing_benchmark.json'
        if prior.is_file():
            saved=load_json(prior)
            if (saved.get('experiment_fingerprint')==self.spec['fingerprint'] and
                saved.get('hardware')==hardware and saved.get('device')==device and
                all(saved.get('models',{}).get(k,{}).get('seconds_per_step',0)>0
                    for k in ('scratch','builtin'))):
                self.stage('timing_benchmark','completed',reused_verified=True,hardware=hardware)
                return saved
        report={'device':device,'policy':'sequential on same idle device',
                'hardware':hardware,'experiment_fingerprint':self.spec['fingerprint'],
                'warmup_steps':5,'timed_steps':20,'models':{}}
        for kind in ('scratch','builtin'):
            proc=self.run_worker(kind,'timing',device)
            for line in proc.stdout:print(f'[timing/{kind}] {line}',end='')
            if proc.wait():raise RuntimeError(f'{kind} timing worker failed')
            report['models'][kind]=load_json(self.base/'timing'/f'{kind}.json')
        a=report['models']['scratch'];b=report['models']['builtin']
        report['scratch_over_builtin_seconds_ratio']=a['seconds_per_step']/b['seconds_per_step']
        report['absolute_trainable_parameter_difference']=abs(a['parameter_count']-b['parameter_count'])
        report['relative_trainable_parameter_difference']=abs(a['parameter_count']-b['parameter_count'])/a['parameter_count']
        write_json(self.base/'analysis/timing_benchmark.json',report)
        self.stage('timing_benchmark','completed')
        return report

    def plots(self):
        self.stage('plots','running')
        import matplotlib.pyplot as plt
        rows={k:load_json(self.base/k/'metrics.json') for k in ('scratch','builtin')}
        timing=load_json(self.base/'analysis/timing_benchmark.json')['models']
        plotdir=self.base/'plots';plotdir.mkdir(exist_ok=True)
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for k in ('scratch','builtin'):
            axes[0,0].plot(range(1,len(rows[k]['train_loss_curve'])+1),rows[k]['train_loss_curve'],label=k)
        axes[0,0].set(title='Matched Q5 training loss',xlabel='Step',ylabel='Cross-entropy');axes[0,0].legend()
        kinds=('scratch','builtin')
        for ax,key,title in [(axes[0,1],'held_out_ppl','Held-out perplexity'),
                             (axes[1,0],'seconds_per_step','Isolated training seconds/step'),
                             (axes[1,1],'parameter_count','Trainable parameters')]:
            src=timing if key=='seconds_per_step' else rows
            ax.bar(kinds,[src[k][key] for k in kinds]);ax.set_title(title)
        fig.tight_layout();fig.savefig(plotdir/'q5_comparison.png',dpi=160);plt.show();plt.close(fig)
        arch=load_json(self.base/'analysis/architecture_comparison.json')
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        axes[0].bar([r['attention_type'] for r in arch['q2']],[r['held_out_ppl'] for r in arch['q2']])
        axes[0].set_title('Historical Q2 scratch attention variants')
        axes[1].bar(('sinusoidal','rope'),(arch['q3']['sinusoidal_ppl'],arch['q3']['rope_ppl']))
        axes[1].set_title('Historical Q3 scratch positions')
        fig.tight_layout();fig.savefig(plotdir/'historical_architecture.png',dpi=160);plt.show();plt.close(fig)
        self.stage('plots','completed',files=[p.name for p in plotdir.glob('*.png')])

    def submission(self,diff):
        self.stage('submission','running')
        timing=load_json(self.base/'analysis/timing_benchmark.json')['models']
        rows={k:load_json(self.base/k/'metrics.json') for k in ('scratch','builtin')}
        metrics={'correctness_max_abs_diff':float(diff),
                 'builtin_held_out_ppl':float(rows['builtin']['held_out_ppl']),
                 'scratch_held_out_ppl':float(rows['scratch']['held_out_ppl']),
                 'builtin_seconds_per_step':float(timing['builtin']['seconds_per_step']),
                 'scratch_seconds_per_step':float(timing['scratch']['seconds_per_step'])}
        if not (0<=metrics['correctness_max_abs_diff']<1e-3 and
                all(math.isfinite(v) and v>0 for v in metrics.values())):
            raise ValueError('Invalid Q5 official metrics')
        sub=self.root/'submissions/q5';sub.mkdir(parents=True,exist_ok=True)
        for name in ('model_builtin.py','q5_runtime.py'):
            target=sub/name;source=self.root/name
            if target.is_file() and sha(target)!=sha(source):
                raise RuntimeError(f'Conflicting existing Q5 submitted source: {target}')
            if not target.is_file():shutil.copy2(source,target)
        final=torch_load(self.base/'builtin/final.pt')
        model=BuiltinDecoderLM(BuiltinLMConfig(**self.cfg));model.load_state_dict(final['model_state'])
        write_torch(sub/'model.pt',cpu_tree(model.state_dict()))
        write_json(sub/'config.json',{'question':'q5',**self.cfg,
            'source_fingerprint':self.spec['fingerprint']})
        write_json(sub/'metrics.json',metrics)
        write_json(self.base/'analysis/metrics_q5.json',metrics)
        logpath=self.root/'analysis_log.json'
        log=load_json(logpath) if logpath.is_file() else {}
        for key,value in metrics.items():log['q5_'+key]=value
        write_json(logpath,log)
        self.stage('submission','completed',path=str(sub))
        return metrics

    def autograder(self):
        self.stage('autograder','running')
        cmd=[sys.executable,str(self.root/'autograder_q5.py'),'--student_dir',str(self.root)]
        result=subprocess.run(cmd,cwd=self.root,text=True,capture_output=True)
        record={'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
                'command':cmd,'utc':now()}
        write_json(self.base/'analysis/autograder_result.json',record)
        print(result.stdout,result.stderr)
        if result.returncode:
            self.stage('autograder','failed',returncode=result.returncode)
            raise RuntimeError('Official Q5 autograder failed; see analysis/autograder_result.json')
        self.stage('autograder','completed',returncode=0)

    def archive(self,partial=False):
        already_complete=self.status['stages']['complete']['state']=='completed'
        if not partial and not already_complete:self.stage('recovery_export','running')
        path=self.root/'q5_all_artifacts.zip';previous=self.root/'q5_all_artifacts.previous.zip'
        prefixes=('artifacts/full/q5','artifacts/smoke/q5','submissions/q5')
        files=[p for prefix in prefixes for p in (self.root/prefix).rglob('*') if p.is_file()
               and '__pycache__' not in p.parts and '.candidate' not in p.name]
        files += [self.root/'model_builtin.py',self.root/'q5_runtime.py']
        manifest={'schema':SCHEMA,'mode':'full','question':'q5','created_utc':now(),
                  'dependency_archives':self.refs,'files':{}}
        for f in sorted(files):
            rel=f.relative_to(self.root).as_posix()
            manifest['files'][rel]={'sha256':sha(f),'size':f.stat().st_size}
        tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
                for f in sorted(files):z.write(f,f.relative_to(self.root).as_posix())
                z.writestr('recovery/integrity_manifest.json',json.dumps(manifest,indent=2))
            with zipfile.ZipFile(tmp) as z:
                assert z.testzip() is None
                required={'artifacts/full/q5/experiment_spec.json'}
                if not partial:
                    required.update(('submissions/q5/model.pt','submissions/q5/config.json',
                        'submissions/q5/metrics.json','artifacts/full/q5/builtin/final.pt',
                        'artifacts/full/q5/scratch/final.pt'))
                assert required<=set(z.namelist())
                for rel,rec in manifest['files'].items():
                    with z.open(rel) as f:
                        h=hashlib.sha256();size=0
                        for chunk in iter(lambda:f.read(1024*1024),b''):
                            h.update(chunk);size+=len(chunk)
                    if h.hexdigest()!=rec['sha256'] or size!=rec['size']:
                        raise ValueError(f'ZIP member failed: {rel}')
            if path.is_file():
                with zipfile.ZipFile(path) as old:
                    old_manifest=json.loads(old.read('recovery/integrity_manifest.json'))
                    if old.testzip() is not None or not old_manifest.get('files'):
                        raise ValueError('Existing Q5 archive is not a known-good predecessor')
                prev_tmp=previous.with_name(previous.name+'.'+uuid.uuid4().hex+'.tmp')
                try:
                    shutil.copy2(path,prev_tmp)
                    if sha(prev_tmp)!=sha(path):raise ValueError('Previous ZIP copy mismatch')
                    os.replace(prev_tmp,previous)
                finally:prev_tmp.unlink(missing_ok=True)
            os.replace(tmp,path)
        finally:tmp.unlink(missing_ok=True)
        if not partial and not already_complete:
            self.stage('recovery_export','completed',archive=str(path),sha256=sha(path))
        return path

    def complete(self):
        mandatory=STAGES[:-1]
        missing=[s for s in mandatory if self.status['stages'][s]['state']!='completed']
        if missing:raise RuntimeError(f'Cannot mark Q5 complete; incomplete stages: {missing}')
        metrics=load_json(self.root/'submissions/q5/metrics.json')
        record={'status':'complete','utc':now(),'autograder':'passed',**metrics,
                'submission':str(self.root/'submissions/q5'),
                'recovery_archive':str(self.root/'q5_all_artifacts.zip'),
                'experiment_fingerprint':self.spec['fingerprint']}
        write_json(self.base/'Q5_COMPLETE.json',record)
        self.stage('complete','completed',**record)
        # Re-export so the verified recovery ZIP also contains the completion marker.
        self.archive()
        return record

def _sampler_batch(dataset,step,batch_size,seed):
    per_epoch=len(dataset)//batch_size
    if per_epoch<1:raise ValueError('Dataset too small for batch')
    epoch=step//per_epoch;offset=step%per_epoch
    g=torch.Generator().manual_seed(seed+epoch)
    order=torch.randperm(len(dataset),generator=g)
    return order[offset*batch_size:(offset+1)*batch_size]

def _rng_state(device):
    return {'cpu':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state(device) if device.type=='cuda' else None,
            'python':random.getstate()}

def _restore_rng(state,device):
    torch.set_rng_state(state['cpu']);random.setstate(state['python'])
    if device.type=='cuda' and state.get('cuda') is not None:
        torch.cuda.set_rng_state(state['cuda'],device)

def _valid_checkpoint(path,spec_hash):
    sidecar=Path(str(path)+'.sha256')
    if not path.is_file() or not sidecar.is_file() or sha(path)!=sidecar.read_text().strip():
        return None
    try:
        payload=torch_load(path)
        if payload['schema']!=SCHEMA or payload['fingerprint']!=spec_hash:
            return None
        for key in ('model_state','optimizer_state','scheduler_state','scaler_state',
                    'completed_steps','tokens_processed','history','evaluation',
                    'cumulative_training_seconds','rng_state','sampler_state'):
            if key not in payload:return None
        return payload
    except Exception:return None

def _checkpoint(run_dir,payload):
    latest=run_dir/'checkpoint.latest.pt';previous=run_dir/'checkpoint.previous.pt'
    candidate=run_dir/('checkpoint.'+uuid.uuid4().hex+'.candidate')
    try:
        torch.save(cpu_tree(payload),candidate)
        # Independent serialized round-trip before promoting.
        read=torch_load(candidate)
        if read['fingerprint']!=payload['fingerprint'] or read['completed_steps']!=payload['completed_steps']:
            raise ValueError('Checkpoint roundtrip mismatch')
        if _valid_checkpoint(latest,payload['fingerprint']) is not None:
            shutil.copy2(latest,previous)
            shutil.copy2(Path(str(latest)+'.sha256'),Path(str(previous)+'.sha256'))
        digest=sha(candidate)
        os.replace(candidate,latest)
        sidecar=Path(str(latest)+'.sha256')
        sidecar_tmp=sidecar.with_name(sidecar.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            sidecar_tmp.write_text(digest)
            os.replace(sidecar_tmp,sidecar)
        finally:sidecar_tmp.unlink(missing_ok=True)
    finally:candidate.unlink(missing_ok=True)

def worker(root,kind,phase,device_name):
    root=Path(root).resolve();spec=load_json(root/'artifacts/full/q5/experiment_spec.json')
    fp=spec['fingerprint'];arch=spec['q1_architecture'];cfg=spec['architecture']
    device=torch.device(device_name)
    if device.type=='cuda':torch.cuda.set_device(device)
    seed=int(spec['seed']);random.seed(seed);torch.manual_seed(seed)
    if device.type=='cuda':torch.cuda.manual_seed(seed)
    scratch_mod,_=scratch_stack(root)
    scratch=scratch_mod.DecoderLM(scratch_config(arch,scratch_mod))
    if kind=='scratch':model=scratch
    else:model=mapped_builtin(scratch,BuiltinDecoderLM(BuiltinLMConfig(**cfg)));del scratch
    model=model.to(device)
    initial_state=cpu_tree(model.state_dict())
    ids=torch_load(root/'artifacts/shared/tokenizer/train_ids.pt').tolist()
    held=load_json(root/'artifacts/full/q5/data_manifest.json')['held_out']
    tok=tokenizer_from_snapshot(root)
    dataset=TokenDataset(ids,cfg['max_seq_len'])
    batch=int(spec['batch_size'])
    amp=device.type=='cuda'
    if phase=='timing':
        run_dir=root/'artifacts/full/q5/timing';run_dir.mkdir(exist_ok=True)
        opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=3e-4)
        scaler=torch.amp.GradScaler('cuda',enabled=amp)
        seconds=[]
        for step in range(25):
            indices=_sampler_batch(dataset,step,batch,seed)
            x,y=zip(*(dataset[int(i)] for i in indices))
            x=torch.stack(x).to(device);y=torch.stack(y).to(device)
            if device.type=='cuda':torch.cuda.synchronize(device)
            start=time.perf_counter()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16 if amp else torch.float32,enabled=amp):
                out=model(x);loss=F.cross_entropy(out.reshape(-1,out.shape[-1]),y.reshape(-1))
            scaler.scale(loss).backward();scaler.step(opt);scaler.update()
            if device.type=='cuda':torch.cuda.synchronize(device)
            if step>=5:seconds.append(time.perf_counter()-start)
        write_json(run_dir/f'{kind}.json',{'device':device_name,'parameter_count':count_trainable(model),
            'seconds_per_step':sum(seconds)/len(seconds),'timed_seconds':sum(seconds),
            'warmup_steps':5,'timed_steps':20,'batch_size':batch,
            'sequence_length':cfg['max_seq_len'],'precision':spec['precision'],
            'optimizer':spec['optimizer'],'includes':'forward+backward+optimizer update'})
        print(f'{kind} isolated timing complete',flush=True);return
    steps=2 if phase=='smoke' else int(spec['steps'])
    run_dir=root/'artifacts'/('smoke' if phase=='smoke' else 'full')/'q5'/kind
    run_dir.mkdir(parents=True,exist_ok=True)
    run_spec={**spec,'kind':kind,'phase':phase,'target_steps':steps,
              'model_source':sha(root/'model_builtin.py') if kind=='builtin' else
              spec['source_hashes']['model_arch.py']}
    run_hash=jhash(run_spec)
    config_path=run_dir/'config.json'
    if config_path.is_file() and load_json(config_path)['run_hash']!=run_hash:
        raise RuntimeError(f'Incompatible existing {kind}/{phase} run')
    write_json(config_path,{'run_hash':run_hash,'run_spec':run_spec})
    metrics_path=run_dir/'metrics.json';final_path=run_dir/'final.pt'
    if metrics_path.is_file() and final_path.is_file():
        try:
            existing=load_json(metrics_path);final=torch_load(final_path)
            if (existing.get('run_hash')==run_hash and final.get('fingerprint')==run_hash
                    and existing['completed_steps']==steps):
                model.load_state_dict(final['model_state'])
                print(f'{kind}/{phase}: validated complete; skip',flush=True);return
        except Exception as exc:
            print(f'{kind}/{phase}: invalid completed artifact, trying checkpoints: {exc}',flush=True)
            model.load_state_dict(initial_state)
    opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=spec['optimizer']['lr'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.0)
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    latest=run_dir/'checkpoint.latest.pt';previous=run_dir/'checkpoint.previous.pt'
    saved=None
    for checkpoint_path in (latest,previous):
        candidate=_valid_checkpoint(checkpoint_path,run_hash)
        if candidate is None:continue
        try:
            model.load_state_dict(candidate['model_state'])
            opt.load_state_dict(candidate['optimizer_state'])
            scheduler.load_state_dict(candidate['scheduler_state'])
            scaler.load_state_dict(candidate['scaler_state'])
        except Exception as exc:
            print(f'{kind}/{phase}: rejected {checkpoint_path.name}: {exc}',flush=True)
            model.load_state_dict(initial_state)
            opt=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=spec['optimizer']['lr'])
            scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.0)
            scaler=torch.amp.GradScaler('cuda',enabled=amp)
            continue
        saved=candidate;break
    completed=0;tokens=0;history=[];evaluations=[];elapsed=0.0
    if saved:
        model.load_state_dict(saved['model_state']);opt.load_state_dict(saved['optimizer_state'])
        scheduler.load_state_dict(saved['scheduler_state']);scaler.load_state_dict(saved['scaler_state'])
        for state in opt.state.values():
            for key,value in state.items():
                if torch.is_tensor(value):state[key]=value.to(device)
        completed=saved['completed_steps'];tokens=saved['tokens_processed']
        history=saved['history'];evaluations=saved['evaluation']
        elapsed=saved['cumulative_training_seconds'];_restore_rng(saved['rng_state'],device)
        print(f'{kind}/{phase}: resumed from step {completed}',flush=True)
    last_save=time.monotonic();start=time.perf_counter()
    def payload(step,seconds):
        return {'schema':SCHEMA,'fingerprint':run_hash,'model_state':cpu_tree(model.state_dict()),
                'optimizer_state':cpu_tree(opt.state_dict()),'scheduler_state':scheduler.state_dict(),
                'scaler_state':scaler.state_dict(),'completed_steps':step,'tokens_processed':tokens,
                'history':history,'evaluation':evaluations,'cumulative_training_seconds':seconds,
                'rng_state':_rng_state(device),'sampler_state':{'scheme':spec['sampler_scheme'],
                    'next_step':step,'seed':seed},'architecture':arch,
                'tokenizer_fingerprint':spec['tokenizer_fingerprint'],
                'dataset_fingerprint':spec['train_ids_hash'],'run_id':f'q5-{phase}-{kind}',
                'checkpoint_schema_version':SCHEMA}
    for step in range(completed,steps):
        indices=_sampler_batch(dataset,step,batch,seed)
        x,y=zip(*(dataset[int(i)] for i in indices))
        x=torch.stack(x).to(device,non_blocking=True);y=torch.stack(y).to(device,non_blocking=True)
        model.train();opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,dtype=torch.float16 if amp else torch.float32,enabled=amp):
            logits=model(x);loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1))
        if not torch.isfinite(loss):raise FloatingPointError(f'{kind} nonfinite loss step {step}')
        scaler.scale(loss).backward();scaler.step(opt);scaler.update();scheduler.step()
        if device.type=='cuda':torch.cuda.synchronize(device)
        tokens+=batch*cfg['max_seq_len'];history.append(float(loss.item()))
        done=step+1
        if phase=='full' and (done%100==0 or done==steps):
            ppl=float(perplexity_of_texts(model,tok,held,device=device,max_len=cfg['max_seq_len']))
            if not math.isfinite(ppl):raise FloatingPointError(f'{kind} nonfinite validation PPL')
            evaluations.append({'step':done,'held_out_ppl':ppl})
            print(f'{kind}/{phase}: {done}/{steps} PPL={ppl:.4f}',flush=True)
        if done%50==0 or time.monotonic()-last_save>=300 or done==steps or (
            phase=='full' and done%100==0):
            seconds=elapsed+time.perf_counter()-start
            _checkpoint(run_dir,payload(done,seconds));last_save=time.monotonic()
    total=elapsed+time.perf_counter()-start
    ppl=float(perplexity_of_texts(model,tok,held,device=device,max_len=cfg['max_seq_len']))
    if not math.isfinite(ppl):raise FloatingPointError(f'{kind} final PPL nonfinite')
    final=payload(steps,total);write_torch(final_path,final)
    metrics={'kind':kind,'phase':phase,'run_hash':run_hash,'held_out_ppl':ppl,
             'parameter_count':count_trainable(model),'completed_steps':steps,
             'tokens_processed':tokens,'training_seconds':total,
             'seconds_per_step':total/steps,'train_loss_curve':history,
             'evaluations':evaluations,'device':device_name,'torch_version':torch.__version__,
             'precision':spec['precision']}
    write_json(metrics_path,metrics)
    print(f'{kind}/{phase}: complete, PPL={ppl:.4f}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=('scratch','builtin'),required=True)
    p.add_argument('--phase',choices=('smoke','full','timing'),required=True)
    p.add_argument('--device',required=True);p.add_argument('--root',required=True)
    a=p.parse_args();worker(a.root,a.worker,a.phase,a.device)
