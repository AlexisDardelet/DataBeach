@echo off
call conda activate video_gpu
cd /d "C:\Users\habib\Documents\GitHub\DataBeach\streamlit_interface"
streamlit run dev_app.py
pause
