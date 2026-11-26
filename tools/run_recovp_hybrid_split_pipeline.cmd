@echo off
cd /d D:\code\Xin\ISRJ_222
D:\code\Xin\ISRJ_222\.venv\Scripts\python.exe -u -m tools.run_recovp_hybrid_split_pipeline --poll-seconds 60 --retry-seconds 60 >> D:\code\Xin\ISRJ_222\evidence\outputs\recovp_hybrid_split_gpu_pipeline\controller.stdout.log 2>&1
