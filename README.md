# 📰 Lazy Economical News Reader

News agencies like Reuters, Yahoo Finance, and others publish dozens of economic articles every day. To stay up to date, readers usually fall into a few types:
- those who spend lots of time reading financial news,
- those who do not read any,
- those who skim headlines or social media and get hype instead of facts,
- and those who read deep technical threads (Slack, Twitter, Substack, etc.) but miss the general economic picture.

At the same time, investing itself has become a trend, which means it’s easy to drown in noise, scattered opinions, and hype cycles. I want a simple way to stay in touch with the **overall economic situation** without getting lost in details.

The aim of this project is to create a short, structured view of what actually matters:

1. **Key economic indicators** — a small set of signals describing the state of the economy.
2. **Industry trends** — sectors move in cycles; tracking them helps understand both opportunities and concentration risks.
3. **Daily news digest** — a compact summary of the major economic events of the day.

A useful weekly example is the [John Hancock Investments’ Weekly Market Recap](https://www.jhinvestments.com/weekly-market-recap#market-moving-news), but my goal is broader and less detailed: a personal system that provides a clearer, day-to-day economic overview and filters out unnecessary noise.

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
