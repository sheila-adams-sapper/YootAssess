I developed this project as an Insight Health Data Science Fellow in June 2018.  The goal was to develop a tool to help doctors improve the accuracy of urinary tract infections (UTI) based on measurements that can be obtained during an office visit.  The minimum viable product associated with this project is called Yoot Assess: Improving UTI Assessment in the Clinic.  The online tool can be temporarily found at www.yootassess.online.  The details outlining the project and model selection and testing can be found in the about.txt file.
The probabilities of urinary tract infection for the assessed patient were developed from a data set published at the following source:

R. Andrew Taylor*, Christopher L. Moore, Kei-Hoi Cheung, Cynthia Brandt
PLoS ONE 13(3): e019408

The data set contained 80,000 observations of adult men and women presenting to the Emergency Department (1 of 4 in the same health system) with UTI-like symptoms between March 2013 and May 2016.  Physicians made diagnostic assessments based on laboratory tests, vital signs, previous healthy history (including medications), and demographic information.  More than 200 such measures were available as features in the data set.  The data also included the physicians’ diagnosis, and the results of the urine culture results, not available to the physician to make an assessment, but provided the ground truth for ultimate diagnosis and a baseline of comparison for empiric versus gold standard culture diagnosis. None of the suspect UTI cases were caused by healthcare associated events, which is important to meet the most basic case of generalizability to community-associated cases that would seek care in a clinic setting.  It does not meet strict generalizability as I touch upon in my summary.

Preprocessing and data cleaning:

The original researchers discretized many of the features, including the vital sign measurements.  I chose to remove these as no code-book was included for the 1-5 scale and my goal was to have interpretable measurements.  Overall, 22% of the presenting patients had true UTI. In many cases, the attending physician did not order certain tests if not warranted, so many of the features contained a majority of ‘not_recorded’ values.  I chose to remove the entire feature if it contained ‘not_recorded’ values greater than 20% of the total for that feature.
‘AbxUTI’ (physician prescription of antibiotics to the patient) was removed as it is correlated to all cases doctor diagnosed as UTI. ‘Ua_bacteria’ was removed as it is the result for the urine culture (‘true_dx’) feature. Physician diagnosis (‘dr_dx’) was not included in the training or test sets of the model.
Dummy variables were created for all string categorical features. All features coded (yes or Yes), (no or No) were encoded as 1,0. 
Three continuous features (pH, specific gravity, and age) were checked for distribution and missing values.  For pH, I replaced NaN values with imputed mean for the outcome category. Specific gravity had 3 physiologically incorrect values, so these observations were dropped.
I used sub-sampling to balance the data set to achieve 50% observations for each outcome type.  This reduced the data set from 80,000 observations to slightly less than 40,000. In subsequent analysis I would compare this strategy to class-weighting.

I looked at the correlations to remove or watch for those with correlations higher than 70%.  In subsequent analysis I would add interaction terms to the model to test associations.

Modeling:

I chose logistic regression in the initial modeling steps to obtain interpretable results that could be used by a physician.  To reduce the feature space, I chose to compare two models: logistic regression with an L1 regularization penalty, and Recursive Feature Elimination (with no regularization).  Hyperparameters were tuned with 3 to 5-fold cross-validation. Model accuracy was tested with 10-fold cross-validation.
The accuracy of both logistic models was similar. However, the model with L1 regularization returned features representing measures already used by physicians in a test called the ‘dipstick’, which is a multi-test urinalysis that can return immediate results.  The RFE model had few of these features and also contained hospital-specific features which would not be generalizable to clinics.  I confirmed the features of the L1 regularization by fitting a random forest model. Examination of the top 20 ranked features showed high similarity to L1 regularization.

Results:

The recall of the logistic regression model was double compared to the doctor assessment (81% vs. 41%).  This is potentially useful in terms of capturing a higher proportion of true UTI cases.  However, in terms of preventing excess antibiotic prescribing by reducing false positives, the model did not do better than the physicians in this scenario, although the overall model accuracy is higher.
In summary, the model can potentially improve same-day assessment of UTI through tests already used by physicians in combination with prior history and demographic information.  However, caution must be used in generalizability to the population of patients presenting to a doctor’s office with UTI symptoms compared with those presenting to an Emergency Department.
In the balanced data set used for the model, 54% of women and 38% of men presenting with UTI-like symptoms had true UTI.  Women comprised 72% of the total samples. Since this observation is consistent with the known higher frequency of UTI in females (excluding healthcare-associated conditions), I also analyzed the data set by extracting the female samples.  The resulting features were highly similar, but the model was much less predictive in terms of recall and accuracy.  The data should be re-analyzed with class-weighting rather than discarding the majority of observations for this scenario.

Sheila Adams-Sapper received her PhD in Infectious Disease from UC Berkeley. She used basic and functional genomics to elucidate the stress-induced transcriptional changes in metabolic pathways of pathogenic bacteria after lethal antibiotic exposure. 




