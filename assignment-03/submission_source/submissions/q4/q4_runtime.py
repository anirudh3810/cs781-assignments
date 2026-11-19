"""Q4 controlled optimizer experiments, two-GPU scheduling, and recovery."""
import gc
import json
import math
import os
import queue
import shutil
import threading
import time
import traceback
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from common import TokenDataset, perplexity_of_texts
from model_arch import LMConfig, DecoderLM
from optim_muon import Muon
from q3_runtime import (atomic_json, atomic_torch, cpu_weights, file_sha256,
                        load_torch, stable_hash, verify_file, weights_hash, StepBatchSampler)

OPTIMIZERS = ("sgd", "rmsprop", "adamw", "muon")
SCHEMA = 1


class BroadMomentum1D(torch.optim.Optimizer):
    """Experimental 1D Muon-style momentum, intentionally without NS."""
    def __init__(self, params, lr, momentum=.95):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.ndim == 2:
                    raise ValueError("BroadMomentum1D only handles non-2D tensors")
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(parameter.grad)
                parameter.add_(buf, alpha=-group["lr"])


class Q4Runtime:
    def __init__(self, *, arch, selected_attention, selected_position, dependencies,
                 train_ids, held_out, tokenizer, grouper, seed, batch_size, seq_len,
                 full_steps, smoke=False):
        self.arch = dict(arch)
        self.selected_attention = selected_attention
        self.selected_position = selected_position
        self.dependencies = dict(dependencies)
        self.semantic_dependencies = {key:value for key,value in dependencies.items()
                                      if not key.endswith("archive_sha256")}
        self.train_ids = [int(value) for value in train_ids]
        self.held_out = list(held_out)
        self.tokenizer = tokenizer
        self.grouper = grouper
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.full_steps = int(full_steps)
        self.smoke = bool(smoke)
        self.mode = "smoke" if self.smoke else "full"
        self.root = Path("artifacts") / self.mode / "q4"
        self.archive = Path("q4_smoke_all_artifacts.zip" if self.smoke else "q4_all_artifacts.zip")
        self.io_lock = threading.RLock()
        self.build_lock = threading.Lock()
        self.last_export = 0.0
        self.archive_seconds = 600.0
        self.initial_hash = None
        self.source_hashes = {}
        self.dataset_size = len(TokenDataset(self.train_ids, self.seq_len))
        if self.dataset_size < self.batch_size:
            raise ValueError("Q4 corpus has fewer than one full training batch")
        if self.selected_attention not in ("mha", "gqa", "mqa") or self.arch["attention_type"] != "standard":
            raise ValueError("Q4 requires the measured standard-attention Q2 winner")
        if self.arch["pos_emb_type"] != self.selected_position:
            raise ValueError("Q4 positional mode differs from measured Q3 selection")

    def build_model(self):
        # No unsafe global RNG reset in parallel workers: CPU fork is serialized.
        with self.build_lock, torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            return DecoderLM(LMConfig(**self.arch)).cpu()

    def setup(self):
        for sub in ("trials", "best", "muon_split_ablation", "analysis", "plots",
                    "source_snapshots", "manifests", "recovery"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        for name in ("a1_tokenizer.py", "rope.py", "attention.py", "model_arch.py",
                     "optim_muon.py", "q3_runtime.py", "q4_runtime.py"):
            active = Path(name)
            if not active.is_file():
                raise FileNotFoundError(f"Q4 source missing: {name}")
            self.source_hashes[name] = file_sha256(active)
            snapshot = self.root / "source_snapshots" / name
            if snapshot.is_file() and file_sha256(snapshot) != self.source_hashes[name]:
                raise RuntimeError(f"Immutable Q4 source snapshot changed: {name}")
            if not snapshot.is_file():
                shutil.copy2(active, snapshot)
        manifest = {"schema": SCHEMA, "mode": self.mode, "architecture": self.arch,
                    "attention": self.selected_attention, "position": self.selected_position,
                    "dependencies": self.semantic_dependencies, "source_hashes": self.source_hashes,
                    "seed": self.seed, "batch_size": self.batch_size,
                    "seq_len": self.seq_len, "full_steps": self.full_steps}
        target = self.root / "manifests" / "experiment.json"
        if target.is_file() and json.loads(target.read_text()) != manifest:
            raise RuntimeError("Existing Q4 experiments have incompatible architecture/data/source")
        if not target.is_file(): atomic_json(target, manifest)
        status_path=self.root/"pipeline_status.json"
        if not status_path.is_file():
            names=("bootstrap","smoke_tests","hpo","hpo_screening","hpo_sgd",
                   "hpo_rmsprop","hpo_adamw","hpo_muon","best_run_selection",
                   "convergence_analysis","lr_sensitivity","muon_split_ablation",
                   "submission","autograder","recovery_export","complete")
            atomic_json(status_path,{"schema":SCHEMA,
                                     "stages":{name:{"status":"pending"} for name in names}})
        initial = self.root / "manifests" / "initial.pt"
        if initial.is_file():
            if not verify_file(initial): raise RuntimeError("Q4 initial state checksum failed")
            state = load_torch(initial)
        else:
            state = cpu_weights(self.build_model())
            atomic_torch(initial, state)
        self.initial_hash = weights_hash(state)
        model = self.build_model(); model.load_state_dict(state, strict=True)
        del model
        return manifest

    def stage(self, name, status, **details):
        path = self.root / "pipeline_status.json"
        with self.io_lock:
            report = json.loads(path.read_text()) if path.is_file() else {"schema":SCHEMA,"stages":{}}
            report["stages"][name] = {"status":status, "updated_utc":datetime.now(timezone.utc).isoformat(), **details}
            atomic_json(path, report)

    def trial_id(self, phase, optimizer, lr, variant):
        rate = f"{lr:.8g}".replace(".", "p").replace("-", "m").replace("+", "")
        return f"{phase}-{optimizer}-{variant}-{rate}"

    def trial_dir(self, phase, optimizer, lr, variant="recommended"):
        if variant == "broad":
            return self.root / "muon_split_ablation" / self.trial_id(phase,optimizer,lr,variant)
        return self.root / "trials" / optimizer / self.trial_id(phase,optimizer,lr,variant)

    def spec(self, phase, optimizer, lr, steps, variant):
        companion_lr = 3e-4 if optimizer == "muon" and variant == "recommended" else None
        hyper = {"sgd":{"momentum":.9}, "rmsprop":{"alpha":.99,"eps":1e-8},
                 "adamw":{"weight_decay":.01},
                 "muon":{"momentum":.95,"ns_steps":5,"companion_adamw_lr":companion_lr}}[optimizer]
        return {"schema":SCHEMA, "id":self.trial_id(phase,optimizer,lr,variant),
                "mode":self.mode, "phase":phase, "optimizer":optimizer, "variant":variant,
                "lr":float(lr), "hyperparameters":hyper, "steps":int(steps),
                "batch_size":self.batch_size, "seq_len":self.seq_len,
                "architecture":self.arch, "initial_state_hash":self.initial_hash,
                "seed":self.seed, "data_seed":self.seed+10000,
                "dependencies":self.semantic_dependencies,
                "source_hashes":self.source_hashes}

    def _prepare(self, spec):
        path = self.trial_dir(spec["phase"],spec["optimizer"],spec["lr"],spec["variant"])
        path.mkdir(parents=True, exist_ok=True)
        config = path / "config.json"
        expected = dict(spec, fingerprint=stable_hash(spec))
        if config.is_file() and json.loads(config.read_text()) != expected:
            raise RuntimeError(f"Incompatible Q4 trial at {path}")
        if not config.is_file():
            with self.io_lock: atomic_json(config, expected)
        return path, expected

    def _optimizers(self, model, spec):
        name, lr, variant = spec["optimizer"], spec["lr"], spec["variant"]
        if name == "sgd": return [torch.optim.SGD(model.parameters(),lr=lr,momentum=.9)], {}
        if name == "rmsprop": return [torch.optim.RMSprop(model.parameters(),lr=lr,alpha=.99,eps=1e-8)], {}
        if name == "adamw": return [torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)], {}
        if variant == "recommended":
            groups = self.grouper(model)
            optimizers = [Muon(groups["muon_params"],lr=lr,momentum=.95,ns_steps=5),
                          torch.optim.AdamW(groups["adamw_params"],lr=3e-4,weight_decay=.01)]
            group_record = {key:groups[key] for key in
                            ("muon_names","adamw_names","muon_count","adamw_count")}
            return optimizers, group_record
        if variant == "broad":
            matrix = [(n,p) for n,p in model.named_parameters() if p.requires_grad and p.ndim==2]
            other = [(n,p) for n,p in model.named_parameters() if p.requires_grad and p.ndim!=2]
            ids = {id(p) for _,p in matrix} | {id(p) for _,p in other}
            assert ids == {id(p) for p in model.parameters() if p.requires_grad}
            assert not {id(p) for _,p in matrix} & {id(p) for _,p in other}
            optimizers = [Muon([p for _,p in matrix],lr=lr,momentum=.95,ns_steps=5),
                          BroadMomentum1D([p for _,p in other],lr=lr,momentum=.95)]
            return optimizers, {"broad_2d_names":[n for n,_ in matrix],
                                "broad_non2d_names":[n for n,_ in other]}
        raise ValueError(variant)

    @staticmethod
    def _state_to_device(optimizer, device):
        for state in optimizer.state.values():
            for key,value in list(state.items()):
                if torch.is_tensor(value): state[key] = value.to(device)

    def _payload(self, spec, model, optimizers, scaler, history, step, elapsed,
                 device, cpu_generator, group_record):
        state = cpu_weights(model)
        sampler = StepBatchSampler(self.dataset_size,self.batch_size,self.seed+10000,step,step)
        return {"schema":SCHEMA, "id":spec["id"], "fingerprint":stable_hash(spec),
                "completed_steps":step, "tokens_processed":step*self.batch_size*self.seq_len,
                "model_state":state, "model_state_hash":weights_hash(state),
                "optimizer_states":[opt.state_dict() for opt in optimizers],
                "scheduler_state":None, "amp_scaler_state":scaler.state_dict(),
                "cpu_rng_state":cpu_generator.get_state(),
                "device_rng_state":torch.cuda.get_rng_state(device).cpu() if device.type=="cuda" else torch.get_rng_state().cpu(),
                "sampler_state":sampler.state(step), "history":history,
                "history_hash":stable_hash(history), "cumulative_seconds":elapsed,
                "group_record":group_record, "spec":spec,
                "saved_utc":datetime.now(timezone.utc).isoformat()}

    def _validate_payload(self, spec, payload):
        if payload.get("schema")!=SCHEMA or payload.get("id")!=spec["id"] or payload.get("fingerprint")!=stable_hash(spec):
            raise ValueError("Incompatible Q4 checkpoint identity/fingerprint")
        step=payload.get("completed_steps")
        if type(step) is not int or not 0<=step<=spec["steps"]: raise ValueError("Q4 checkpoint step invalid")
        hist=payload.get("history")
        if not isinstance(hist,list) or len(hist)!=step or [row["step"] for row in hist]!=list(range(1,step+1)):
            raise ValueError("Q4 checkpoint history discontinuity")
        if payload.get("history_hash")!=stable_hash(hist): raise ValueError("Q4 history digest mismatch")
        if payload.get("tokens_processed")!=step*self.batch_size*self.seq_len: raise ValueError("Q4 token count mismatch")
        sampler=StepBatchSampler(self.dataset_size,self.batch_size,self.seed+10000,step,step)
        if payload.get("sampler_state")!=sampler.state(step): raise ValueError("Q4 sampler cursor mismatch")
        if weights_hash(payload["model_state"])!=payload.get("model_state_hash"):
            raise ValueError("Q4 checkpoint tensor digest mismatch")
        if len(payload.get("optimizer_states",[]))!=(2 if spec["optimizer"]=="muon" else 1):
            raise ValueError("Q4 optimizer-state count mismatch")
        if not isinstance(payload.get("cumulative_seconds"),(int,float)) or not math.isfinite(payload["cumulative_seconds"]):
            raise ValueError("Q4 elapsed time invalid")
        model=self.build_model(); model.load_state_dict(payload["model_state"],strict=True)
        opts,_=self._optimizers(model,spec)
        for opt,state in zip(opts,payload["optimizer_states"]): opt.load_state_dict(state)
        del model,opts
        return payload

    def _load_checkpoint(self, path, spec):
        errors=[]
        for candidate in (path/"checkpoint.latest.pt",path/"checkpoint.previous.pt"):
            if not candidate.is_file(): continue
            try:
                if not verify_file(candidate): raise ValueError("checksum mismatch")
                result=self._validate_payload(spec,load_torch(candidate))
                if candidate.name.endswith("previous.pt"):
                    print(f"[{spec['id']}] restored previous-known-good checkpoint")
                return result
            except Exception as exc: errors.append(f"{candidate.name}: {exc}")
        if errors: raise RuntimeError("No valid Q4 checkpoint: "+"; ".join(errors))
        return None

    def _save_checkpoint(self, path, spec, model, optimizers, scaler, history,
                         step, elapsed, device, cpu_generator, group_record):
        payload=self._payload(spec,model,optimizers,scaler,history,step,elapsed,
                              device,cpu_generator,group_record)
        latest,previous=path/"checkpoint.latest.pt",path/"checkpoint.previous.pt"
        with self.io_lock:
            candidate=latest.with_name(latest.name+"."+uuid.uuid4().hex+".candidate")
            torch.save(payload,candidate)
            self._validate_payload(spec,load_torch(candidate))
            latest_valid = False
            if latest.is_file() and verify_file(latest):
                try:
                    self._validate_payload(spec,load_torch(latest))
                    latest_valid = True
                except Exception:
                    latest_valid = False
            if latest_valid:
                old=previous.with_name(previous.name+"."+uuid.uuid4().hex+".tmp")
                shutil.copy2(latest,old)
                if file_sha256(old)!=file_sha256(latest): raise IOError("Q4 previous-checkpoint copy changed")
                os.replace(old,previous)
                atomic_json(previous.with_name(previous.name+".sha256.json"),
                            {"sha256":file_sha256(previous),"size":previous.stat().st_size})
            digest=file_sha256(candidate)
            os.replace(candidate,latest)
            atomic_json(latest.with_name(latest.name+".sha256.json"),
                        {"sha256":digest,"size":latest.stat().st_size})
            atomic_json(path/"status.json",{"status":"running","completed_steps":step,
                       "checkpoint_sha256":digest,"device":str(device),"id":spec["id"]})

    def _terminal(self, path, spec):
        metrics_path,status_path=path/"metrics.json",path/"status.json"
        if not (metrics_path.is_file() and status_path.is_file()): return None
        metrics=json.loads(metrics_path.read_text()); status=json.loads(status_path.read_text())
        if metrics.get("fingerprint")!=stable_hash(spec) or status.get("id")!=spec["id"]:
            raise RuntimeError(f"Q4 trial has incompatible existing result: {path}")
        if status.get("status")=="diverged" and metrics.get("state")=="diverged": return metrics
        if status.get("status")!="complete" or metrics.get("state")!="complete": return None
        if (metrics.get("completed_steps")!=spec["steps"] or
            len(metrics.get("loss_curve",[]))!=spec["steps"] or
            len(metrics.get("time_curve",[]))!=spec["steps"]): return None
        final=path/"final.pt"
        if not verify_file(final) or file_sha256(final)!=metrics.get("final_file_sha256"):
            print(f"Q4 completed final file invalid at {path}; reconstructing from verified checkpoint")
            return None
        if weights_hash(load_torch(final))!=metrics.get("final_state_hash"):
            print(f"Q4 completed final tensors invalid at {path}; reconstructing from verified checkpoint")
            return None
        return metrics

    def run_trial(self, optimizer_name, lr, steps, *, phase, device_name, position,
                  variant="recommended"):
        if optimizer_name not in OPTIMIZERS or variant not in ("recommended","broad"):
            raise ValueError("Unknown Q4 optimizer/variant")
        if variant=="broad" and optimizer_name!="muon": raise ValueError("Only Muon has a broad ablation")
        spec=self.spec(phase,optimizer_name,float(lr),int(steps),variant)
        path,_=self._prepare(spec)
        result=self._terminal(path,spec)
        if result is not None:
            print(f"Q4 validated completed trial skipped: {spec['id']}")
            return result
        prior=self._load_checkpoint(path,spec)
        device=torch.device(device_name)
        model=self.build_model()
        model.load_state_dict(prior["model_state"] if prior else load_torch(self.root/"manifests"/"initial.pt"),strict=True)
        model=model.to(device)
        optimizers,group_record=self._optimizers(model,spec)
        try: scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
        except (AttributeError,TypeError): scaler=torch.cuda.amp.GradScaler(enabled=device.type=="cuda")
        generator=torch.Generator(device="cpu").manual_seed(self.seed+20000)
        history,step,elapsed_base=[],0,0.0
        if prior:
            for opt,state in zip(optimizers,prior["optimizer_states"]):
                opt.load_state_dict(state); self._state_to_device(opt,device)
            scaler.load_state_dict(prior["amp_scaler_state"])
            generator.set_state(prior["cpu_rng_state"])
            if device.type=="cuda": torch.cuda.set_rng_state(prior["device_rng_state"],device)
            else: torch.set_rng_state(prior["device_rng_state"])
            history=list(prior["history"]); step=prior["completed_steps"]
            elapsed_base=float(prior["cumulative_seconds"])
        elif device.type=="cuda":
            torch.cuda.set_rng_state(torch.Generator(device=device).manual_seed(self.seed+30000).get_state(),device)
        else:
            torch.manual_seed(self.seed+30000)
        dataset=TokenDataset(self.train_ids,self.seq_len)
        sampler=StepBatchSampler(self.dataset_size,self.batch_size,self.seed+10000,step,steps)
        loader=DataLoader(dataset,batch_sampler=sampler,num_workers=0,
                          pin_memory=device.type=="cuda",generator=generator)
        started=time.monotonic(); last_ckpt=step; last_ckpt_time=started
        last_safe_step=step
        bar=tqdm(total=steps,initial=step,position=position,leave=True,
                 desc=f"Q4 {optimizer_name} lr={lr:g} {phase} {device}",dynamic_ncols=True)
        try:
            with self.io_lock:
                atomic_json(path/"status.json",{"status":"running","completed_steps":step,
                           "device":str(device),"id":spec["id"],"resumed":bool(prior)})
            for x,y in loader:
                x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
                for opt in optimizers: opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                    logits=model(x)
                    loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),y.reshape(-1))
                if not torch.isfinite(loss): raise FloatingPointError("nonfinite Q4 training loss")
                scaler.scale(loss).backward()
                last_safe_step=-1  # mid-update must never be checkpointed
                for opt in optimizers: scaler.step(opt)
                scaler.update()
                if device.type=="cuda": torch.cuda.synchronize(device)
                step+=1; last_safe_step=step
                elapsed=elapsed_base+time.monotonic()-started
                history.append({"step":step,"loss":float(loss.item()),
                                "cumulative_seconds":elapsed,
                                "tokens_processed":step*self.batch_size*self.seq_len,
                                "epoch":sampler.state(step)["epoch"]})
                with self.io_lock:
                    atomic_json(path/"history.json",{"history":history,"completed_steps":step,
                                                   "cumulative_seconds":elapsed})
                due=(step==steps or step-last_ckpt>=50 or time.monotonic()-last_ckpt_time>=300 or
                     step==max(1,steps//2))
                if due:
                    self._save_checkpoint(path,spec,model,optimizers,scaler,history,step,
                                          elapsed,device,generator,group_record)
                    last_ckpt=step; last_ckpt_time=time.monotonic()
                    self.export_archive(force=False)
                session=max(time.monotonic()-started,1e-9)
                speed=(step-(prior["completed_steps"] if prior else 0))/session
                bar.update(1)
                bar.set_postfix({"state":"RESUMED" if prior else "NEW", "loss":f"{sum(r['loss'] for r in history[-20:])/min(20,len(history)):.2f}",
                                 "lr":f"{lr:.1e}","tokens":step*self.batch_size*self.seq_len,
                                 "step/s":f"{speed:.2f}","tok/s":f"{speed*self.batch_size*self.seq_len:.0f}",
                                 "session":f"{session:.0f}s","total":f"{elapsed:.0f}s",
                                 "eta":f"{(steps-step)/max(speed,1e-9):.0f}s","ckpt":last_ckpt})
            if device.type=="cuda": torch.cuda.synchronize(device)
            model.eval()
            ppl=float(perplexity_of_texts(model,self.tokenizer,self.held_out,
                                          device=str(device),max_len=self.arch["max_seq_len"]))
            if not math.isfinite(ppl) or ppl<1: raise FloatingPointError("nonfinite held-out PPL")
            elapsed=elapsed_base+time.monotonic()-started
            with self.io_lock:
                final=path/"final.pt"; atomic_torch(final,cpu_weights(model))
                metrics={"schema":SCHEMA,"id":spec["id"],"state":"complete",
                         "mode":self.mode,"phase":phase,"optimizer":optimizer_name,
                         "variant":variant,"lr":lr,"hyperparameters":spec["hyperparameters"],
                         "seed":self.seed,"data_seed":self.seed+10000,"steps_requested":steps,
                         "completed_steps":step,"held_out_ppl":ppl,
                         "loss_curve":[r["loss"] for r in history],
                         "time_curve":[r["cumulative_seconds"] for r in history],
                         "history":history,"total_seconds":elapsed,
                         "mean_seconds_per_step":history[-1]["cumulative_seconds"]/step,
                         "training_seconds":history[-1]["cumulative_seconds"],
                         "evaluation_seconds":elapsed-history[-1]["cumulative_seconds"],
                         "tokens_processed":step*self.batch_size*self.seq_len,
                         "device":str(device),"checkpoint_path":str(path/"checkpoint.latest.pt"),
                         "final_path":str(final),"final_file_sha256":file_sha256(final),
                         "final_state_hash":weights_hash(load_torch(final)),
                         "fingerprint":stable_hash(spec),"initial_state_hash":self.initial_hash,
                         "group_record":group_record}
                atomic_json(path/"metrics.json",metrics)
                atomic_json(path/"status.json",{"status":"complete","id":spec["id"],
                                                  "completed_steps":step,"held_out_ppl":ppl})
            print(f"Q4 {spec['id']} complete on {device}; PPL={ppl:.4f}, duration={elapsed:.1f}s")
            return metrics
        except FloatingPointError as exc:
            elapsed=elapsed_base+time.monotonic()-started
            metrics={"schema":SCHEMA,"id":spec["id"],"state":"diverged","mode":self.mode,
                     "phase":phase,"optimizer":optimizer_name,"variant":variant,"lr":lr,
                     "hyperparameters":spec["hyperparameters"],"seed":self.seed,
                     "steps_requested":steps,"completed_steps":last_safe_step,
                     "held_out_ppl":float("inf"),"loss_curve":[r["loss"] for r in history],
                     "time_curve":[r["cumulative_seconds"] for r in history],
                     "history":history,"total_seconds":elapsed,
                     "tokens_processed":last_safe_step*self.batch_size*self.seq_len,
                     "device":str(device),"fingerprint":stable_hash(spec),"error":str(exc)}
            with self.io_lock:
                atomic_json(path/"metrics.json",metrics)
                atomic_json(path/"status.json",{"status":"diverged","id":spec["id"],
                                                      "error":str(exc),"completed_steps":last_safe_step})
            print(f"Q4 {spec['id']} diverged; excluded from best selection: {exc}")
            return metrics
        except BaseException as exc:
            try:
                if last_safe_step==step and step>0:
                    self._save_checkpoint(path,spec,model,optimizers,scaler,history,step,
                        elapsed_base+time.monotonic()-started,device,generator,group_record)
            except Exception as checkpoint_error:
                print(f"Q4 emergency checkpoint failed: {checkpoint_error}")
            with self.io_lock:
                atomic_json(path/"status.json",{"status":"failed","id":spec["id"],
                            "completed_steps":step,"error":f"{type(exc).__name__}: {exc}",
                            "traceback":traceback.format_exc()})
            raise
        finally:
            bar.close(); del model,optimizers,loader,dataset
            gc.collect()
            if device.type=="cuda": torch.cuda.empty_cache()

    def run_jobs(self, jobs):
        jobs=list(jobs)
        if not jobs: return []
        if not self.smoke and torch.cuda.device_count()<2:
            raise RuntimeError("Full Q4 HPO requires Kaggle 2 x T4 GPUs")
        devices=["cuda:0","cuda:1"] if torch.cuda.device_count()>=2 else ["cpu"]
        pending=queue.Queue()
        for position,job in enumerate(jobs): pending.put((position,job))
        results={}; failures={}; lock=threading.Lock()
        start_barrier=threading.Barrier(2) if len(devices)==2 and len(jobs)>=2 else None
        def worker(slot,device):
            first=True
            while True:
                try: position,job=pending.get_nowait()
                except queue.Empty: break
                try:
                    if first and start_barrier is not None: start_barrier.wait(timeout=60)
                    first=False
                    result=self.run_trial(job["optimizer"],job["lr"],job["steps"],
                        phase=job["phase"],device_name=device,position=slot,
                        variant=job.get("variant","recommended"))
                    with lock: results[position]=result
                except BaseException as exc:
                    with lock: failures[position]=f"{type(exc).__name__}: {exc}"
                finally: pending.task_done()
        with ThreadPoolExecutor(max_workers=len(devices),thread_name_prefix="q4gpu") as pool:
            futures=[pool.submit(worker,i,device) for i,device in enumerate(devices)]
            for future in futures: future.result()
        self.export_archive(force=True)
        if failures:
            raise RuntimeError(f"Q4 trial failures (other GPU progress retained): {failures}")
        return [results[i] for i in range(len(jobs))]

    def select_best(self, optimizer, trial_results):
        eligible=[result for result in trial_results if result["optimizer"]==optimizer and
                  result["phase"]=="full" and result["state"]=="complete" and
                  math.isfinite(result["held_out_ppl"])]
        if len(eligible)<2: raise RuntimeError(f"Q4 {optimizer} has fewer than two successful full LR trials")
        winner=min(eligible,key=lambda row:(row["held_out_ppl"],row["total_seconds"],row["lr"]))
        bestdir=self.root/"best"/optimizer; bestdir.mkdir(parents=True,exist_ok=True)
        source=Path(winner["final_path"])
        if not verify_file(source): raise RuntimeError(f"Selected Q4 model invalid: {source}")
        destination=bestdir/"model.pt"
        if not verify_file(destination) or weights_hash(load_torch(destination))!=winner["final_state_hash"]:
            atomic_torch(destination,load_torch(source))
        record={"optimizer":optimizer,"trial_id":winner["id"],"lr":winner["lr"],
                "held_out_ppl":winner["held_out_ppl"],"model_path":str(destination),
                "model_sha256":file_sha256(destination),"trial_final_path":str(source),
                "steps":winner["completed_steps"],"total_seconds":winner["total_seconds"],
                "fingerprint":winner["fingerprint"]}
        atomic_json(bestdir/"selection.json",record)
        return record

    def validate_archive(self,path):
        with zipfile.ZipFile(path) as archive:
            manifest=json.loads(archive.read("recovery/integrity_manifest.json"))
            if manifest.get("question")!="q4" or manifest.get("mode")!=self.mode:
                raise RuntimeError("Q4 recovery ZIP identity mismatch")
            for name,info in manifest["files"].items():
                digest,size=__import__("hashlib").sha256(),0
                with archive.open(name) as handle:
                    for chunk in iter(lambda:handle.read(1024*1024),b""):
                        digest.update(chunk);size+=len(chunk)
                if digest.hexdigest()!=info["sha256"] or size!=info["size"]:
                    raise RuntimeError(f"Q4 recovery member mismatch: {name}")
        return manifest

    def export_archive(self,force=False):
        """Create a crash-safe *compact* recovery ZIP.

        Completed trials do not need optimizer checkpoints for recovery, and
        screening trials do not need trained model weights after their metrics
        have been persisted.  Excluding those redundant files keeps the archive
        small enough for Kaggle and avoids the multi-GB archive explosion that
        can kill the session after HPO has already finished.
        """
        with self.io_lock:
            now=time.monotonic()
            if not force and now-self.last_export<self.archive_seconds:
                return self.archive if self.archive.is_file() else None

            def include_in_recovery(path):
                # Never include temporary/candidate files.
                if ".candidate" in path.name or ".tmp" in path.name:
                    return False
                if path == self.root/"recovery"/"latest_export.json":
                    return False

                parent = path.parent
                status_path = parent/"status.json"
                config_path = parent/"config.json"
                status = {}
                config = {}
                try:
                    if status_path.is_file():
                        status = json.loads(status_path.read_text())
                except Exception:
                    pass
                try:
                    if config_path.is_file():
                        config = json.loads(config_path.read_text())
                except Exception:
                    pass

                completed = status.get("status") == "complete"

                # A completed trial can always be reconstructed from final.pt +
                # config/history/metrics. Its optimizer checkpoints are only
                # useful while the trial is incomplete.
                if completed and path.name in ("checkpoint.latest.pt",
                                               "checkpoint.previous.pt"):
                    return False

                # Screening trials are used only for LR-ranking evidence.
                # Their metrics/history/config are sufficient; their final model
                # weights are never used by Q4.4-Q4.7.
                if completed and config.get("phase") == "screening" and path.name == "final.pt":
                    return False

                return True

            files=[path for path in self.root.rglob("*")
                   if path.is_file() and include_in_recovery(path)]

            submission=Path("submissions/q4")
            if not self.smoke and submission.is_dir():
                files += [path for path in submission.rglob("*") if path.is_file()]
            files=sorted(set(files),key=str)

            manifest={"schema":SCHEMA,"question":"q4","mode":self.mode,
                      "created_utc":datetime.now(timezone.utc).isoformat(),
                      "architecture_fingerprint":stable_hash(self.arch),
                      "dependencies":self.dependencies,
                      "source_hashes":self.source_hashes,
                      "archive_policy":{
                          "completed_checkpoints_omitted":True,
                          "screening_final_weights_omitted":True,
                          "reason":"compact Kaggle-safe recovery archive"
                      },
                      "files":{path.as_posix():{"sha256":file_sha256(path),
                                                "size":path.stat().st_size}
                               for path in files}}

            # Keep at most one old archive, but delete an older .previous copy
            # *before* constructing the candidate. This caps peak disk usage at
            # roughly current + candidate instead of current + previous +
            # candidate (+ a temporary copied previous archive).
            previous=self.archive.with_name(
                self.archive.name.replace(".zip",".previous.zip"))
            if previous.is_file():
                try:
                    previous.unlink()
                except OSError:
                    pass

            candidate=self.archive.with_name(
                self.archive.name+"."+uuid.uuid4().hex+".candidate")
            with zipfile.ZipFile(candidate,"w",
                                 compression=zipfile.ZIP_DEFLATED,
                                 allowZip64=True) as archive:
                for path in files:
                    archive.write(path,arcname=path.as_posix())
                archive.writestr(
                    "recovery/integrity_manifest.json",
                    json.dumps(manifest,indent=2))

            self.validate_archive(candidate)

            # Candidate is already validated. Preserve the current archive by
            # an atomic rename (no multi-GB shutil.copy2), then promote candidate.
            if self.archive.is_file():
                try:
                    self.validate_archive(self.archive)
                    os.replace(self.archive,previous)
                except Exception as exc:
                    print("Existing Q4 ZIP invalid; replacing it:",exc)
                    try:
                        self.archive.unlink(missing_ok=True)
                    except Exception:
                        pass

            os.replace(candidate,self.archive)
            digest=file_sha256(self.archive)
            atomic_json(self.root/"recovery"/"latest_export.json",
                        {"archive":str(self.archive.resolve()),
                         "sha256":digest,
                         "size":self.archive.stat().st_size,
                         "compact":True})
            self.last_export=now
            print(f"Validated compact Q4 recovery ZIP: {self.archive} "
                  f"({digest[:12]}..., "
                  f"{self.archive.stat().st_size/2**20:.1f} MiB)")
            return self.archive
