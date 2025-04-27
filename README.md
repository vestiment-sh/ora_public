<p align = "center">
  <img src = "images/ora_lobster_title.png" alt = "logo" width = "250"/>
</p>

## Ora is a fine-tuned DistilBERT model trained on teacher evaluations from the Richard Gilder Graduate School's MAT ESRP Program.

![demo](images/ora.gif)

### About
Ora, Italian for _*now*_, is a fine-tuned DistilBERT model trained on mentor evaluations from the Richard Gilder Graduate School’s Master of Arts in Teaching Earth Science Residency Program (MAT ESRP). This program prepares aspiring STEM educators through rigorous scientific and pedagogical training, combined with hands-on teaching experience in high-needs schools. Mentors who have successfully completed the program assess residents’ performance based on guidelines set by the Council for the Accreditation of Educator Preparation (CAEP). Residents are graded on 18 factors, including communication, ethics, and instruction, with scores categorized as Unsatisfactory, Basic, Proficient, or Accomplished. These ratings are numerically converted on a 1–4 scale, and mentors’ notes are used to evaluate the program’s effectiveness and identify areas for improvement. Once evaluations are completed, they are manually entered into FileMaker Pro and submitted to the CAEP board every few years. As expected, this process is time-consuming and prone to human error.

Ora, short for _*Observation Rubric Analysis*_, originated as my capstone project analyzing the program’s performance over the past three cohorts. I examined overall metrics, identified outliers, and created visuals to highlight key patterns. I additionally experimented with several transformer models to gain insights into grading interpretation, sentiment, and the summaries of mentor notes. The Ora model automates the assessment of prospective educators based solely on observational data, providing instant scores and averages. Trained on prior evaluations, it identifies key words and sequences that distinguish an Unsatisfactory performance from an Accomplished one.
