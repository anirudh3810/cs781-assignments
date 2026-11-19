# Anirudh Bankhede — CS781 Assignment 5

Roll number: **240131**  
Email: **anirudhb24@iitk.ac.in**

## Kaggle setup

1. Create a Kaggle notebook and use **File → Import Notebook** to import
   `CS781_Assignment_5_Questions.ipynb`. Required code, preference data, support files,
   and graders are embedded; separate Python-file uploads are unnecessary.
2. In **Settings**, select **GPU T4 x2**, enable **Internet**, and choose
   **Session persistence → Files**. Apply these settings before starting execution.
3. Use **Run All** for an interactive run, or **Save Version → Save & Run All** for
   a background version. Use only one active run against the same artifact directory.
4. The notebook checks dependencies and both GPUs first, then runs Q1–Q6 in order,
   generates the analytical answers and figures, runs the graders, and packages valid results.

The expected final file is **CS781-A5-Anirudh-Bankhede-240131.zip**. It appears in
`/kaggle/working/` and in `assignment5_artifacts/deliverables/`.

## Checkpoints and recovery

Training checkpoints save every **100 updates or 900 seconds**, with recovery copies.
Completed compatible stages are reused; Q5 interventions save after each head.
All progress is under `/kaggle/working/assignment5_artifacts/` by default.

Kaggle controls session duration and file persistence. Enable Files persistence before
the session stops and save/download outputs before deleting or discarding a session.
If the session ends before completion, restore its files and use Run All again.
If you attach a prior saved output as notebook input, set `RESTORE_ARTIFACTS_FROM`
in the first configuration cell to its `assignment5_artifacts` directory. The bootstrap
copies it to writable storage before resuming.

Both GPUs execute independent Q1/Q2/Q3 experiments and Q5 head interventions.
Baseline stages with dependencies, CPU numerical checks, plotting and packaging have
different utilization. Full utilization of both GPUs at every instant is not a valid
expectation for these workloads. No fixed session duration or target metric is guaranteed.
If a required grokking, parity, reconstruction or grading check fails, the notebook
retains its measurements and stops final packaging instead of claiming success.

The final archive includes a notebook with measured analytical answers, source modules,
official model artifacts, figures, reports, preference data, and your student details.
It excludes internal optimizer checkpoints and the Hugging Face download cache.
