# 📰 Lazy Economical News Reader

News agencies like Reuters, Yahoo Finance, and others publish dozens of economic articles every day.  
To stay up to date, readers have to scroll constantly and go through long summaries, which means there are usually two types of people:
- Those who spend lots of time reading financial news
- Those who do not read any

I’m looking for a **compromise solution** — a short, smart summary that highlights **only the key economic areas and major events** that made the headlines each day (or week).

A great weekly example is [John Hancock Investments’ Weekly Market Recap](https://www.jhinvestments.com/weekly-market-recap#market-moving-news).  
My goal is to build a shorter **daily version** of a similar recap.

## 🎯 Project Goals

This repository contains **Google Colab notebooks** used in early trials to verify that the following tasks are feasible:

1. **Parsing news pages** from major financial sources  
2. **Identifying key daily topics** and grouping related articles using GPT-based algorithms  

## 🧠 Future Plans

Eventually, the project will:
- Collect and store parsed news in an SQL database in real time  
- Summarize and categorize economic headlines automatically  
- Provide daily digests

## ⚙️ Current Stage
- Environment: Google Colab  
- Version control: GitHub  
- Next step: Transition to AWS for automated real-time parsing and data storage
- Containerization: Plan to use **Docker** for environment consistency and scalable deployment  
- Final ready code will also be maintained and updated in this GitHub repository
