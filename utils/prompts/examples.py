GENERAL_Q_EXAMPLES = [
        { # entry 18: general rephrasing
            "q_orig": "Characteristic of venous blood flow of lower limb in duplex Doppler is?",
            "ans_ops": ["Monophasic", "Biphasic", "Triphasic", "Non phasic"],
            "q_upd": "What is characteristic of venous blood flow in the lower limb on duplex Doppler?",
        },
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
        { # entry 88: not including answer options in the updated question
            "q_orig": "Polydactyly, craniosynostosis, Late closure of fontanelles is a feature of:",
            "ans_ops": ["Apert's syndrome", "Crouton's syndrome", "Pierre robin syndrome", "Down' syndrome"],
            "q_upd": "Polydactyly, craniosynostosis, and late closure of fontanelles are features of which syndrome?",
        },
        { # entry 4: generic question; "which"
            "q_orig": "Low insulin to glucagon ratio is seen in all of these except:",
            "ans_ops": ["Glycogen synthesis", "Glycogen breakdown", "Gluconeogenesis", "Ketogenesis"],
            "q_upd": "Which of the following is not associated with a low insulin-to-glucagon ratio?",
        },
        { # entry 11: keeping true / false structure
            "q_orig": "A second-year PG resident tells you to perform an ABG of a patient. All of the following are true about performing an ABG except:",
            "ans_ops": ['Before performing the ABG, syringe should be loaded with 0.3 cc of heparin', 'Normal pH, HCO. and PCO, levels may not indicate absence of an acid-base imbalance', "A different site should be tried if modified Allen's test is negative", 'Radial aery is the preferred site'],
            "q_upd": "When performing an ABG analysis, which of the following is not true?",
        },
        { # entry 33: not removing the provided answer options from the original question
            "q_orig": "Steps of intubation - arrange in sequence:- a. Head extension and flexion of neck b. Introduction of laryngoscope c. Inflation of cuff d. Check breath sounds with stethoscope e. fixation of the tube to prevent dislodgement",
            "ans_ops": ["ABCDE", "DBCEA", "ACBED", "CBAED"],
            "q_upd": "What are the steps of intubation in sequence, given:\nA. Head extension and flexion of neck\nB. Introduction of laryngoscope\nC. Inflation of cuff\nD. Check breath sounds with stethoscope\nE. fixation of the tube to prevent dislodgement?",
        },
        # { # entry 79: importance of preserving original question context
        #   "q_orig": "Child of Vasanthi was weaned from breast milk on the 5th day and was given sugarcane juice the child developed hypoglycemia and hepatomegaly biochemical examination showed hypophosphatemia and enzyme deficiencies–reducing substances in urine. The child is probably suffering from which of the following enzyme deficiencies –",
        #   "ans_ops": ["Fructokinase", "Aldolase B", "Glucose 6 Phosphatase", "Beta galactosidase"],
        #   "q_upd": "A child weaned from breast milk on day 5 and given sugarcane juice develops hypoglycemia, hepatomegaly, hypophosphatemia, and reducing substances in urine. Which enzyme deficiency is most likely?",
        # },
    ]
OP_LIST_EXAMPLES = [
        {  # entry 567: preserving the original question context while improving clarity
            "q_orig": "A 20 years female has hepatosplenomegaly, fever, pallor and generalized lymphadenopathy. Lab test useful for diagnosis is/are -a) ESRb) Electrophoresisc) Parasite detection in aspirated) ELISAe) Routine haemogram",
            "ans_ops": ["ab", "abc", "ae", "bde"],
            "q_upd": "What is the laboratory test that would be most useful for diagnosing a 20-year-old female presenting with hepatosplenomegaly, fever, pallor, and generalized lymphadenopathy?\na) ESR,\nb) Electrophoresis, \nc) Parasite detection in aspirate,\nd) ELISA,\ne) Routine haemogram.",
        },
        {  # entry 582: preserving the original question context while improving clarity
            "q_orig": "Persistent vomiting in G.O.O. causes -a) Hyponatremic hyperchloremia occurb) Hypernatremia without ↓ed Cl- alkalosisc) Hypokalemic metabolic alkalosisd) Paradoxical aciduria",
            "ans_ops": ["ab", "abc", "ae", "bde"],
            "q_upd": "Persistent vomiting in gastric outlet obstruction (G.O.O.) causes which of the following?\na) Hyponatremic hyperchloremia\nb) Hypernatremia without decreased chloride, with alkalosis\nc) Hypokalemic metabolic alkalosis\nd) Paradoxical aciduria",
        },
        {  # entry 9: preserving the original question context while improving clarity
            "q_orig": "Characteristics of Remifentanyl – a) Metabolised by plasma esteraseb) Short half lifec) More potent than Alfentanyld) Dose reduced in hepatic and renal diseasee) Duration of action more than Alfentanyl",
            "ans_ops": ["ab", "bc", "abc", "bcd"],
            "q_upd": "Which of the following are characteristics of remifentanil?\na) Metabolised by plasma esterase\nb) Short half life\nc) More potent than Alfentanil\nd) Dose reduced in hepatic and renal disease\ne) Duration of action more than Alfentanil",
        },
    ]
QUESTION_EXAMPLES = [
    *GENERAL_Q_EXAMPLES,
]
ANS_OP_EXAMPLES = [
    *GENERAL_Q_EXAMPLES,
    { # entry 8105: no need to give the answer or change the question word
        "q_orig": "Which of the following is the ego-expansion of JSY?",
        "ans_ops": ["Janani Sampoorna Yojana", "Janani Samridhi Yojana", "Janani Swarojgar Yojana", "Janani Surakshan Yojana"],
        "q_upd": "Which of the following correctly expands the acronym JSY?",
    },
    { # entry 36040: no need to expand the abbreviation
        "q_orig": "What is meant by PET scan?",
        "ans_ops": ["Positive Emission Tomography", "Positron Emission Tomography", "Positron Energy Tomography", "Positive Energy Tomography"],
        "q_upd": "What is meant by a PET scan?",
    },
]
QUESTION_OP_LIST_EXAMPLES = [
    # *GENERAL_Q_EXAMPLES,
    *OP_LIST_EXAMPLES,
]

def format_question_examples(examples: list[dict], no_ans_op: bool = False) -> list[str]:
    """
    Format question examples for prompting. If no_ans_op is True, the answer options will not
    be included in the formatted output.
    :param examples: A list of examples, where each example is a dict with keys 'q_orig', 'ans_ops', and 'q_upd'.
    :param no_ans_op: A boolean flag indicating whether to include answer options in the formatted output (default: False).
    :return: A list of formatted example strings.
    """
    if no_ans_op:
        return [
            "Ex {}.\n- INPUT -\nQUESTION: {}\n\n- OUTPUT-\n{}\n".format(i, ex['q_orig'], ex['q_upd'])
            for i, ex in enumerate(examples, 1)
        ]
    return [
        ("Ex {}.\n- INPUT -\nQUESTION: {}\nPOSSIBLE ANSWERS:\n- {}\n\n"
         "- OUTPUT-\n{}\n").format(i, ex['q_orig'], '\n- '.join(ex['ans_ops']),                                                                         ex['q_upd'])
        for i, ex in enumerate(examples, 1)
    ]


ANSWER_EXAMPLES = [
    {
        "q": "What is the principle of Chinese medicine?",
        "ans_ops": ["Yang", "Vin", "Both", "None"],
        "ans_ops_upd": ["Yang", "Yin", "Both", "None"],
    },
    {
        "q": "What visual field defect is typically produced by a temporal lobe tumor?",
        "ans_ops": ["Crossed upper Quadrantanopia", "Uncrossed upper Quadrantanopia", "crossed lower Quadrantanopia", "Uncrossed lower Quadrantanopia"],
        "ans_ops_upd": ["Crossed upper quadrantanopia", "Uncrossed upper quadrantanopia", "Crossed lower quadrantanopia", "Uncrossed lower quadrantanopia"],
    },
    { # dev 182867
        "q": "Multiple canals in mandibular premolars are seen in?",
        "ans_ops": ["Africas", "Caucians", "Not Recalled", "Not Recalled"],
        "ans_ops_upd": ["Africans", "Caucasians", "Not recalled", "Not recalled"],
    },
    { # test 188562
        "q": "If the dentist concludes you don't have a cavity, when you really have one or more?",
        "ans_ops": ["Type 2 error", "Type 1 error", "Type 3 error", "one of the above"],
        "ans_ops_upd": ["Type 2 error", "Type 1 error", "Type 3 error", "one of the above"],
    },
    { # test 188729
        "q": "Which type of ZnPO4 cement is fine-grained, having film thickness of 25 micrometers?",
        "ans_ops": ["Type 1", "Type 2", "Type 3", "Type 4"],
        "ans_ops_upd": ["Type 1", "Type 2", "Type 3", "Type 4"],
    },
]


def wrap_options(ans_ops: list[str]) -> str:
    """
    Wrap answer options with their corresponding labels (A, B, C, D).
    :param ans_ops: A list of answer option strings.
    :return: A formatted string with each option labeled and separated by newlines.
    """
    options = ["A", "B", "C", "D"]
    return '\n'.join([f"{op}. {ans}" for op, ans in zip(options, ans_ops)])


FORMATTED_ANSWER_EXAMPLES = [
        "Ex {}.\n- INPUT -\nQuestion: {}\nAnswer options:\n{}\n\n- OUTPUT-\n{}\n".format(
            i, ex["q"], wrap_options(ex["ans_ops"]), wrap_options(ex["ans_ops_upd"])
        )
        for i, ex in enumerate(ANSWER_EXAMPLES, 1)
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
        "Ex {}.\n- INPUT -\n{}\n\n- OUTPUT-\n{}\n".format(i, ex['exp_orig'], ex['exp_upd'])
        for i, ex in enumerate(EXPLANATION_EXAMPLES, 1)
]

FORMATTED_EXAMPLES = {
    "question_op_list_disappeared_ids": format_question_examples(QUESTION_OP_LIST_EXAMPLES, no_ans_op=True),
    "question_ans_op_appeared_ids": format_question_examples(ANS_OP_EXAMPLES, no_ans_op=True),
    "answer": FORMATTED_ANSWER_EXAMPLES,
    "question": format_question_examples(QUESTION_EXAMPLES),
    "explanation": FORMATTED_EXPLANATION_EXAMPLES,
}

def get_examples_for_task(task: str) -> list:
    """
    Retrieve the formatted examples for a given task.
    The function checks if the task name starts or ends with any of the keys in the
    FORMATTED_EXAMPLES dictionary and returns the corresponding examples.
    If no matching examples are found, it returns an empty list.
    :param task: The name of the task for which to retrieve examples
                 (e.g., "question", "answer", "explanation").
    :return: A list of formatted example strings for the specified task,
             or an empty list if no examples are found.
    """
    for name, examples in FORMATTED_EXAMPLES.items():
        if task.startswith(name) or task.endswith(name):
            print("Using examples for task:", name)
            return examples
    print("No examples found for task:", task, "among available tasks:", list(FORMATTED_EXAMPLES.keys()))
    return []