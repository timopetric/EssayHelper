import classla
from transformers import AutoTokenizer

from tqdm import tqdm
import logging
import argparse
import sherpa
import pathlib
from datetime import datetime

from LektorReader import extract_sentence_and_w_forms_dict_from_line
from SlobertaMCD import SlobertaMCD
from helpers import *

from EssayHelper import EssayHelper



logging.basicConfig(
    # %(asctime)s  %(funcName)s:
    format='%(levelname)-8s [%(filename)s:%(lineno)3d]  %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)
# coloredlogs.install(level='DEBUG', logger=logger)


def iter_lektor_lines(tokenizer_sloberta, max_batch_len=10000, only_one_batch=False, test_mode=False, eval_params: EvalParams=None):
    print(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(f"$$$$ LEKTOR PARAMS: $$$$$$$$$$$$$$$")
    print(f"$$$$     max_batch_len: {max_batch_len}")
    print(f"$$$$     only_one_batch: {only_one_batch}")
    print(f"$$$$     test_mode: {test_mode}")
    print(f"$$$$     eval_params: {eval_params}")
    print(f"$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        
    filter_dict = {}
    if eval_params:
        for n_from, _ in eval_params.number_from_to:
            key = f"n{n_from}"
            if key not in filter_dict:
                filter_dict[key] = 0
            else:
                filter_dict[key] += 1
        for c_from, _ in eval_params.case_from_to:
            key = f"c{c_from}"
            if key not in filter_dict:
                filter_dict[key] = 0
            else:
                filter_dict[key] += 1

    number_case4, number_case4_key  = 0, "c4"
    number_case2, number_case2_key  = 0, "c2"
    number_number3, number_number3_key = 0, "n3"
    number_number2, number_number2_key = 0, "n2"

    with open(PTH_LEKTOR_OUT_TEST if test_mode else PTH_LEKTOR_OUT_TRAIN, "r") as lektor_in:
        # for input_text in sentence_examples:
        essay_text = ""
        num_sentences = 0
        for input_line_num, input_line in enumerate(lektor_in.readlines()):
            input_text, w_forms_dict = extract_sentence_and_w_forms_dict_from_line(input_line)

            # print(filter_dict, [w_forms_dict[key] for key in filter_dict], sum([w_forms_dict[key] for key in filter_dict]), eval_params.num_wrong_words)
            if test_mode and eval_params and sum([w_forms_dict[key] for key in filter_dict]) < eval_params.num_wrong_words:
                # filter out sentences without needed cases and numbers
                continue
                
            if input_text.count("\n") == 1 \
                    and input_text[-1] == "\n" \
                    and get_num_tokens_sloberta_for_sent(input_text, tokenizer_sloberta) <= tokenizer_sloberta.model_max_length:  # (sloberta accepts max 512 tokens per input)

                number_case4 += w_forms_dict[number_case4_key]
                number_case2 += w_forms_dict[number_case2_key]
                number_number3 += w_forms_dict[number_number3_key]
                number_number2 += w_forms_dict[number_number2_key]
                
                essay_text += input_text
                num_sentences += 1
                if num_sentences+1 >= max_batch_len:
                    print("-------- lektor counts: ----------")
                    print("number of words with case4:", number_case4)
                    print("number of words with case2:", number_case2)
                    print("number of words with number3:", number_number3)
                    print("number of words with number2:", number_number2)
                    print("----------------------------------")
                    yield essay_text
                    
                    essay_text = ""
                    num_sentences = 0
                    if only_one_batch:
                        return
                    
        print("-------- lektor counts: ----------")
        print("number of words with case4:", number_case4)
        print("number of words with case2:", number_case2)
        print("number of words with number3:", number_number3)
        print("number of words with number2:", number_number2)
        print("----------------------------------")
        yield essay_text


def get_num_tokens_sloberta_for_sent(sent, tokenizer_sloberta):
    tokens = tokenizer_sloberta(sent, add_special_tokens=True, verbose=True).get("input_ids")
    assert tokens and isinstance(tokens, list) and len(tokens) > 1, f"tokenizer out is weird: {tokens}"
    return len(tokens)


def sherpa_optimization_thresholds(essay: EssayHelper, study_name, thresholds, sherpa_port=8880, params_to_train=None):
    print(f"Doing a shepra parameter optimization for thresholds.")
        
    def get_params(params_to_train=None):
        parameters = []
        
        if params_to_train:
            for name in params_to_train:
                params = [0, 0.15] if "case" in name else [0, 0.60] if "num" in name else [0, 1.0]
                print("WARNING: case and num training ranges are hardcoded: ", params)
                parameters.append(
                    sherpa.Continuous(name, params)
                )
            param_keys = params_to_train
            
        return parameters, param_keys

    def get_threshold_params_dict(thresholds, fields_to_train):
        all_fields = set(Thresholds._fields)
        remaining_fields = list(all_fields - set(fields_to_train))
        
        thresholds_dict = {
            field: 1 if field.startswith(("case", "number")) else thresholds.__getattribute__(field)
            for field in
            remaining_fields
        }
            
        return thresholds_dict

    def filter_params(trail_params, param_keys):
        return {
            i: trail_params[i]
            for i in 
            trail_params
            if i in param_keys
        }

    def save_progress(study, study_out_dir, settings_str):
        output_dir = os.path.join(PTH_PARAM_OPTIMIZATION_STUDIES, study_out_dir, settings_str)
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        study.save(output_dir=output_dir)

    # num_generations = 100
    # population_size = 40
    # num_iterations = num_generations * population_size
    num_iterations = 1000
    
    # algorithm = sherpa.algorithms.GridSearch(num_grid_points=num_iterations)
    # algorithm = sherpa.algorithms.PopulationBasedTraining(num_generations, population_size=population_size)
    algorithm = sherpa.algorithms.RandomSearch(max_num_trials=num_iterations)
    parameters, _ = get_params(params_to_train)
    study = sherpa.Study(
        parameters=parameters,
        algorithm=algorithm,
        lower_is_better=False,
        disable_dashboard=False,
    )
    
    max_correction_perc = 0
    for inx, trial in enumerate(tqdm(study, total=num_iterations, desc=f"Finding best params for '{study_name}'")):
        params = filter_params(trial.parameters, params_to_train)
        
        thresholds_dict_other = get_threshold_params_dict(thresholds, params_to_train)

        thr = Thresholds(
            **params,
            # **{"case42": 0.001, "number32": 0.00001},  # "number32": 0.00001, 
            **thresholds_dict_other
        )

        try:
            essay.thresholds = thr  # TODO: update every sentence/word
            num_sentences, num_wrong_words, conf_matrix, accuracy, precision, recall, f1_score, num_all_words \
                = essay.eval_calculate_word_form_predictions_from_suggested_word_replacements(thr, multiple_passes=True)

            # print(f"##################### {params} #####################################")
            # essay.print_corrected_essay(only_corrected=False)
            objective = f1_score
            
            study.add_observation(
                trial=trial,
                objective=objective,
                context={
                    **params,
                    **conf_matrix._asdict(),
                    "num_sentences": num_sentences,
                    "num_wrong_words": num_wrong_words,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                }
            )

        except Exception as e:
            logger.exception(f"Could not calculate_word_form_predictions_from_suggested_word_replacements: {e}")
            continue
        finally:
            study.finalize(trial=trial)
            t_best = study.get_best_result()
            t_obj_perc = t_best.get("Objective", -1.0)

            if t_obj_perc > max_correction_perc or inx == num_iterations-1:
                print("\n\n\nBest trial so far:", t_best, "\n\n\n")

                date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
                t_id = t_best.get("Trial-ID", -1)
                t_gen = t_best.get("generation", -1)
                essay_num_sents = num_sentences
                study_settings_str = f"{date_str}_tId{t_id:04d}_tGen{t_gen:03d}_tObj{t_obj_perc:6.3f}_eSents{essay_num_sents}"
                save_progress(study, study_name, study_settings_str)

                max_correction_perc = t_obj_perc

    print(f"Finished study.\n{study.get_best_result()}")


def main():
    # ##################### ARGPARSER #####################
    # parser = argparse.ArgumentParser(description="SloHelper program.")
    # # parser.add_argument("-sp", "--sherpa_port", help="sherpa dashbord port", default=8880, type=int, required=True)
    # # parser.add_argument("-epf", "--eval_params_from", help="eval params slicing from", default=0, type=int, required=True)
    # # parser.add_argument("-ept", "--eval_params_to", help="eval params slicing to", default=36, type=int, required=True)
    # # parser.add_argument("-ttp", "--time_to_process_init", help="minutes it takes to init the essay", default=10, type=int, required=True)
    # # parser.add_argument("-nww", "--num_wrong_words", help="sherpa dashbord port", default=1, type=int, required=True)
    # # parser.add_argument("-nls", "--num_lektor_sentences", help="number of lektor sentences", default=1000, type=int, required=True)
    # # parser.add_argument("-sn", "--study_name", help="study name sherpa", default="default_study", type=str, required=True)
    
    # args = parser.parse_args()
    # sherpa_port = args.sherpa_port
    # # num_wrong_words = args.num_wrong_words
    # # arg_study_name = args.study_name
    # eval_params_from = args.eval_params_from
    # eval_params_to = args.eval_params_to
    # # time_to_process_init = args.time_to_process_init
    

    ##################### READY THE PIPELINES #####################
    try:
        cl_pipeline = classla.Pipeline('sl', processors="tokenize,pos,lemma", use_gpu=True, verbose=False)
    except Exception as e:
        if "Try to download the model again" in e.__str__():
            classla.download("sl")
            cl_pipeline = classla.Pipeline('sl', processors="tokenize,pos,lemma", use_gpu=True, verbose=False)


    cl_pipeline_pretok = classla.Pipeline('sl', processors="tokenize,pos,lemma", tokenize_pretokenized=True, use_gpu=True, verbose=False)
    tokenizer_sloberta = AutoTokenizer.from_pretrained("EMBEDDIA/sloberta", verbose=True)
    sloberta_mcd = SlobertaMCD(tokenizer=tokenizer_sloberta, cuda_device=0, verbose=False)

    
    eval_params_list = [
        EvalParams(case_from_to=[(2,4)],
                   number_from_to=[(2,3)], num_wrong_words=0),
        EvalParams(case_from_to=[(2,4)],
                   number_from_to=[(2,3)], num_wrong_words=1),
        EvalParams(case_from_to=[(2,4)],
                   number_from_to=[(2,3)], num_wrong_words=2),
        EvalParams(case_from_to=[(2,4)],
                   number_from_to=[(2,3)], num_wrong_words=3),
        
        # EvalParams(case_from_to=[(2,4)],
        #            number_from_to=[], num_wrong_words=0),
        # EvalParams(case_from_to=[(2,4)],
        #            number_from_to=[], num_wrong_words=1),
        # EvalParams(case_from_to=[(2,4)],
        #            number_from_to=[], num_wrong_words=2),
        # EvalParams(case_from_to=[(2,4)],
        #            number_from_to=[], num_wrong_words=3),
        
        # EvalParams(case_from_to=[],
        #            number_from_to=[(2,3)], num_wrong_words=0),
        # EvalParams(case_from_to=[],
        #            number_from_to=[(2,3)], num_wrong_words=1),
        # EvalParams(case_from_to=[],
        #            number_from_to=[(2,3)], num_wrong_words=2),
        # EvalParams(case_from_to=[],
        #            number_from_to=[(2,3)], num_wrong_words=3),        
    ]
    

    thresholds_prelearned_option = 0
    test_mode = True
    
    for eval_params in eval_params_list:
        print("\n\n\n\n")
        print("////////////////////////////////////////////////////////////////////")
        print("/////////////////// new eval parameter being processed /////////////")
        print("////////////////////////////////////////////////////////////////////")
        for mcd_on in [False, True]: 
            debug_dt_processing_took = datetime.now()

            for essay_text in iter_lektor_lines(tokenizer_sloberta, max_batch_len=3000, only_one_batch=True, test_mode=test_mode, eval_params=None if test_mode else eval_params):
                print("num sentences from lektor loaded:", len(essay_text.split("\n")))
                    
                essay = EssayHelper(
                    essay_text,
                    eval_params=eval_params,
                    cuda_device=0,
                    sherpa_optimizations_todo=[], # ["classla_batch", "sloberta_batch"]
                    cl_pipeline=cl_pipeline,
                    cl_pipeline_pretok=cl_pipeline_pretok,
                    tokenizer_sloberta=tokenizer_sloberta,
                    mcd_on=mcd_on,
                    thresholds_prelearned_option=thresholds_prelearned_option,  # 0 = both, 1 = case, 2 = number
                    sloberta_mcd=sloberta_mcd,
                    multi_pass=True,
                    verbose=False
                )
                
            # # ! uncomment below to eval
            verbose=False
            num_sentences, num_wrong_words, conf_matrix, accuracy, precision, recall, f1_score, num_all_words \
                = essay.eval_calculate_word_form_predictions_from_suggested_word_replacements(verbose=verbose, print_conf_metrices=True)
            
            timedelta_took = datetime.now()-debug_dt_processing_took
            print( "##############################################################")
            print(f"####### Eval params:         {eval_params}")
            print(f"####### Evaluating with MCD: {mcd_on}")
            print(f"####### Processing took:     {timedelta_took} or {timedelta_took.total_seconds()} seconds")
            print( "##############################################################")
            print(f"\t{conf_matrix}")
            print("\tnum_all_words:", num_all_words)
            print("\tnum_wrong_words:", num_wrong_words)
            print("\tnum_sentences:", num_sentences)
            print(f"\taccuracy:  {accuracy*100:7.3f}%")
            print(f"\tprecision: {precision*100:7.3f}%")  # tocnost
            print(f"\trecall:    {recall*100:7.3f}%")     # priklic
            print(f"\tf1 score:  {f1_score*100:7.3f}%")
            print( "######################## eval end ############################")
            print( "##############################################################")
            # ! eval end





            # # ! uncomment below to optimize threshold weights
            # # moved to sherpa_optimization_thresholds
            # params_to_train = [f"case{ct}{cf}" for cf, ct in eval_params.case_from_to]
            # params_to_train += [f"number{nt}{nf}" for nf, nt in eval_params.number_from_to]
            # # params_to_train += ["direct_replacement_if_same_lemma", "skip_if_first_sugg_below"]

            # arg_study_name = get_eval_params_str(eval_params)

            # sherpa_optimization_thresholds(essay, arg_study_name, essay.thresholds, params_to_train=params_to_train)  #, sherpa_port=sherpa_port)
            # # ! optimize threshold weights end
        

if __name__ == "__main__":
    main()
