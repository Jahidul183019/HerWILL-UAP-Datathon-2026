# Competition reference

Summary of the competition Terms & Conditions and Kaggle data requirements. Refer to the official rules for updates.

- [Competition](https://www.kaggle.com/competitions/HerWILL-Safe-Social-Media-Datathon-2026)
- [Rules and competition data terms](https://www.kaggle.com/competitions/HerWILL-Safe-Social-Media-Datathon-2026/rules#7-competition-data)
- [Data](https://www.kaggle.com/competitions/HerWILL-Safe-Social-Media-Datathon-2026/data)
- [Organizers' code submission form](https://forms.gle/vERYNZeuupTYS3RM7)

## Modeling and data

- Use only the provided competition dataset; external datasets, scraped data, and additional labeled data are prohibited unless organizers explicitly allow them.
- Public libraries are allowed. Pretrained models are allowed provided no external training data is used; seek organizer clarification for any uncertain model or training setup.
- The dataset cannot be shared, published, or used outside this competition.
- Work must be original, reproducible, and explainable. Document preprocessing, model architecture, hyperparameters, training, and evaluation.
- Evaluation uses Macro F1 over the three classes. The private leaderboard determines final leaderboard ranking.
- The public leaderboard uses approximately 40% of the test data. Final results use the other 60% (the private leaderboard), so final standings may differ. Subset membership is not disclosed.
- Use local validation to guide model selection; treat the public score as partial feedback and avoid tuning repeatedly to it. The 40%/60% split describes Kaggle's test evaluation, not our local training/validation split.

## Required deliverables

1. Submit predictions as a CSV on Kaggle before the deadline, matching `sample_submission.csv` (`id,y_pred`).
2. Submit the complete `.ipynb` or `.py` for the best result using the organizers' form. Code can be submitted only once.
3. Select up to two submissions for final private leaderboard evaluation.

Both predictions and source code are mandatory. Check the exact deadline and daily submission allowance on Kaggle before submitting.

## Shortlisting and presentation

The top-10 shortlist considers private leaderboard performance and technical evaluation. Shortlisted teams must give a presentation of at most five minutes covering the problem, methodology, model rationale, results, challenges, and improvements. Judges select the top five based on performance, presentation, and ability to explain the work.
