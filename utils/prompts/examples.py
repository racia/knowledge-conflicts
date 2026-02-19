QUESTION_EXAMPLES = [
        { # entry 12: specific question word addition
            "q_orig": "Respiratory rhythm generation center is located at:",
            "ans_ops": ["Dorsal respiratory group", "Pre-Botzinger complex", "Ventral respiratory neurons", "Pneumotaxic center"],
            "q_upd": "Where is the respiratory rhythm generation center located?",
        },
        { # entry 9: context preservation
            "q_orig": "A blue newborn presents with cyanosis. The X-ray chest reveals oligemic lung fields and a normal-sized heart. Most likely diagnosis is –",
            "ans_ops": ["Ebstein's anomaly", "Pulmonary atresia", "Transposition of great arteries", "Tetralogy of fallot"],
            "q_upd": "A newborn presents with cyanosis (appearing blue). Chest X-ray reveals oligemic lung fields and a normal-sized heart. What is the most likely diagnosis?",
        },
        { # entry 18: general rephrasing
            "q_orig": "Characteristic of venous blood flow of lower limb in duplex Doppler is?",
            "ans_ops": ["Monophasic", "Biphasic", "Triphasic", "Non phasic"],
            "q_upd": "What is characteristic of venous blood flow in the lower limb on duplex Doppler?",
        },
        { # entry 4: generic question; "which"
            "q_orig": "Low insulin to glucagon ratio is seen in all of these except:",
            "ans_ops": ["Glycogen synthesis", "Glycogen breakdown", "Gluconeogenesis", "Ketogenesis"],
            "q_upd": "Which of the following is not associated with a low insulin-to-glucagon ratio?",
        },
        { # entry 33: not removing the provided answer options from the original question
            "q_orig": "Steps of intubation - arrange in sequence:- a. Head extension and flexion of neck b. Introduction of laryngoscope c. Inflation of cuff d. Check breath sounds with stethoscope e. fixation of the tube to prevent dislodgement",
            "ans_ops": ["ABCDE", "DBCEA", "ACBED", "CBAED"],
            "q_upd": "What are the steps of intubation in sequence, given: A. Head extension and flexion of neck; B. Introduction of laryngoscope; C. Inflation of cuff; D. Check breath sounds with stethoscope; E. fixation of the tube to prevent dislodgement?",
        },
        { # entry 11: keeping true / false structure
            "q_orig": "A second-year PG resident tells you to perform an ABG of a patient. All of the following are true about performing an ABG except:",
            "ans_ops": ['Before performing the ABG, syringe should be loaded with 0.3 cc of heparin', 'Normal pH, HCO. and PCO, levels may not indicate absence of an acid-base imbalance', "A different site should be tried if modified Allen's test is negative", 'Radial aery is the preferred site'],
            "q_upd": "When performing an arterial blood gas (ABG) analysis, which of the following is not true?",
        },
        { # entry 88: not including answer options in the updated question
            "q_orig": "Polydactyly, craniosynostosis, Late closure of fontanelles is a feature of:",
            "ans_ops": ["Apert's syndrome", "Crouton's syndrome", "Pierre robin syndrome", "Down' syndrome"],
            "q_upd": "Polydactyly, craniosynostosis, and late closure of fontanelles are features of which syndrome?",
        },
        # { # entry 79: importance of preserving original question context
        #   "q_orig": "Child of Vasanthi was weaned from breast milk on the 5th day and was given sugarcane juice the child developed hypoglycemia and hepatomegaly biochemical examination showed hypophosphatemia and enzyme deficiencies–reducing substances in urine. The child is probably suffering from which of the following enzyme deficiencies –",
        #   "ans_ops": ["Fructokinase", "Aldolase B", "Glucose 6 Phosphatase", "Beta galactosidase"],
        #   "q_upd": "A child weaned from breast milk on day 5 and given sugarcane juice develops hypoglycemia, hepatomegaly, hypophosphatemia, and reducing substances in urine. Which enzyme deficiency is most likely?",
        # },
    ]

FORMATTED_QUESTION_EXAMPLES = [
        ("Ex {}.\n- Input -\nQuestion: {}\nAnswer options:\n{}\n"
         "- Output-\n{}").format(i, ex['q_orig'], '\n'.join(ex['ans_ops']),                                                                         ex['q_upd'])
        for i, ex in enumerate(QUESTION_EXAMPLES, 1)
    ]


EXPLANATION_EXAMPLES = [
    {
        "exp_orig": "Chronic urethral obstruction because of urinary calculi, prostatic hyperophy, tumors, normal pregnancy, tumors, uterine prolapse or functional disorders cause hydronephrosis which by definition is used to describe dilatation of renal pelvis and calculus associated with progressive atrophy of the kidney due to obstruction to the outflow of urine Refer Robbins 7yh/9,1012,9/e. P950",
        "exp_upd": "Chronic obstruction to urinary outflow, such as that caused by urinary calculi, prostatic hypertrophy, tumors, pregnancy, uterine prolapse, or functional outflow disorders, can lead to hydronephrosis. Hydronephrosis is defined as dilatation of the renal pelvis and calyces, associated with progressive atrophy of the renal parenchyma, resulting from persistent obstruction of urine flow.",
    },
    {
        "exp_orig": "Ans. (c) Vitamin B12 Ref: Harrison's 19th ed. P 640* Vitamin B12 (Cobalamin) is synthesized solely by microorganisms.* In humans, the only source for humans is food of animal origin, e.g., meat, fish, and dairy products.* Vegetables, fruits, and other foods of nonanimal origin doesn't contain Vitamin B12 .* Daily requirements of vitamin Bp is about 1-3 pg. Body stores are of the order of 2-3 mg, sufficient for 3-4 years if supplies are completely cut off.",
        "exp_upd": "Vitamin B12 (cobalamin) is synthesized solely by microorganisms. In humans, the only significant dietary sources are foods of animal origin such as meat, fish, and dairy products; fruits, vegetables, and other nonanimal foods do not naturally contain vitamin B12. The daily requirement is approximately 1–3 micrograms, while body stores are about 2–3 milligrams, which are sufficient for roughly 3–4 years even if dietary intake ceases completely.",
    },
    {
        "exp_orig": "Ans. A 45 yo male with DM presents with polyuria polydipsia weight loss. Random sugar 350 mg/dl. HbA1c 10.2%. Most likely diagnosis:",
        "exp_upd": "A 45-year-old male with diabetes mellitus presents with polyuria, polydipsia, and weight loss. Random blood sugar is 350 mg/dL, and HbA1c is 10.2%, consistent with uncontrolled hyperglycemia.",
    },
    {
        "exp_orig": "Hydronephrosis refers to dilation of the renal pelvis and calyces due to obstruction of urine flow, leading to parenchymal atrophy if prolonged. Robbins Basic Pathology, 9th ed., p. 950.",
        "exp_upd": "Hydronephrosis refers to dilation of the renal pelvis and calyces due to obstruction of urine flow, leading to parenchymal atrophy if prolonged. Robbins Basic Pathology, 9th ed., p. 950.",
    },
]

FORMATTED_EXPLANATION_EXAMPLES = [
        "Ex {}.\n- Input -\n{}\n- Output-\n{}\n".format(i, ex['exp_orig'], ex['exp_upd'])
        for i, ex in enumerate(EXPLANATION_EXAMPLES, 1)
]

FORMATTED_EXAMPLES = {
    "question": FORMATTED_QUESTION_EXAMPLES,
    "explanation": FORMATTED_EXPLANATION_EXAMPLES,
}