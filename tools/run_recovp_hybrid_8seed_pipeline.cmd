@echo off
cd /d D:\code\Xin\ISRJ_222
D:\Anaconda\python.exe -u -m tools.run_recovp_hybrid_8seed_pipeline --poll-seconds 60 --retry-seconds 60 >> D:\code\Xin\ISRJ_222\evidence\outputs\recovp_hybrid_8seed_pipeline\controller.stdout.log 2>&1
