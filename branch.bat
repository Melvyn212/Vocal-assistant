@echo off

mkdir assistant-vocal\asr\models
mkdir assistant-vocal\nlu\models
mkdir assistant-vocal\dialogue_management\models
mkdir assistant-vocal\nlg\models
mkdir assistant-vocal\tts\models
mkdir assistant-vocal\tests

copy nul assistant-vocal\asr\asr.py > nul 2>&1
copy nul assistant-vocal\nlu\nlu.py > nul 2>&1
copy nul assistant-vocal\dialogue_management\dialogue.py > nul 2>&1
copy nul assistant-vocal\nlg\nlg.py > nul 2>&1
copy nul assistant-vocal\tts\tts.py > nul 2>&1
copy nul assistant-vocal\main.py > nul 2>&1
