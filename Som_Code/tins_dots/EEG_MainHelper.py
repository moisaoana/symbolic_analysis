from Som_Code.tins_dots.Plots_Generator import Alignment, PlotsGenerator


class EEG_MainHelper:

    @staticmethod
    def take_minimum_window_from_trials_start(all_trials, trials_lengths):
        minimum_trial_length = min(trials_lengths)
        for cnt, trial in enumerate(all_trials):
            all_trials[cnt].trial_data = trial.trial_data[0:minimum_trial_length]

    @staticmethod
    def take_minimum_window_from_trials_end(all_trials, trials_lengths):
        minimum_trial_length = min(trials_lengths)
        for cnt, trial in enumerate(all_trials):
            trial_len = len(all_trials[cnt].trial_data)
            all_trials[cnt].trial_data = trial.trial_data[trial_len-minimum_trial_length:trial_len]



    @staticmethod
    def main_with_psi1(list_freq_by_response, list_freq_by_stimulus, list_freq_by_visibility, som, eegDataProcessor, pathLeft, pathRight, params, no_samples, weighted=False):
        response_PSIs_for_all_colors_matrix_array = PlotsGenerator.computePSIByGroup(list_freq_by_response,
                                                                                     som)

        stimulus_PSIs_for_all_colors_matrix_array = PlotsGenerator.computePSIByGroup(list_freq_by_stimulus,
                                                                                     som)

        visibility_PSIs_for_all_colors_matrix_array = PlotsGenerator.computePSIByGroup(list_freq_by_visibility,
                                                                                       som)


        PlotsGenerator.groupByResponseWithPsiUsingBMU(eegDataProcessor.trials,
                                                      som,
                                                      response_PSIs_for_all_colors_matrix_array,
                                                      pathLeft,
                                                      params,
                                                      no_samples,
                                                      alignment=Alignment.LEFT,
                                                      weighted=weighted)

        PlotsGenerator.groupByResponseWithPsiUsingBMU(eegDataProcessor.trials,
                                                      som,
                                                      response_PSIs_for_all_colors_matrix_array,
                                                      pathRight,
                                                      params,
                                                      no_samples,
                                                      alignment=Alignment.RIGHT,
                                                      weighted=weighted)

        # -------------------------------------------------------------------

        PlotsGenerator.groupByStimulusWithPsiUsingBMU(eegDataProcessor.trials,
                                                      som,
                                                      stimulus_PSIs_for_all_colors_matrix_array,
                                                      pathLeft,
                                                      params,
                                                      no_samples,
                                                      alignment=Alignment.LEFT,
                                                      weighted=weighted)

        PlotsGenerator.groupByStimulusWithPsiUsingBMU(eegDataProcessor.trials,
                                                      som,
                                                      stimulus_PSIs_for_all_colors_matrix_array,
                                                      pathRight,
                                                      params,
                                                      no_samples,
                                                      alignment=Alignment.RIGHT,
                                                      weighted=weighted)
        # -------------------------------------------------------------------

        PlotsGenerator.groupByVisibilityWithPsiUsingBMU(eegDataProcessor.trials,
                                                        som,
                                                        visibility_PSIs_for_all_colors_matrix_array,
                                                        pathLeft,
                                                        params,
                                                        no_samples,
                                                        alignment=Alignment.LEFT,
                                                        weighted=weighted)

        PlotsGenerator.groupByVisibilityWithPsiUsingBMU(eegDataProcessor.trials,
                                                        som,
                                                        visibility_PSIs_for_all_colors_matrix_array,
                                                        pathRight,
                                                        params,
                                                        no_samples,
                                                        alignment=Alignment.RIGHT,
                                                        weighted=weighted)

    @staticmethod
    def main_with_psi2(list_freq_by_response, list_freq_by_stimulus, list_freq_by_visibility, som, eegDataProcessor, pathLeft, pathRight, params, no_samples):
        response_PSIs_for_all_colors_matrix_array = PlotsGenerator.computePSIByGroupMethod2(list_freq_by_response,
                                                                                            som)
        stimulus_PSIs_for_all_colors_matrix_array = PlotsGenerator.computePSIByGroupMethod2(list_freq_by_stimulus,
                                                                                            som)

        visibility_PSIs_for_all_colors_matrix_array = PlotsGenerator.computePSIByGroupMethod2(list_freq_by_visibility,
                                                                                              som)


        PlotsGenerator.groupByResponseWithPsiUsingBMU(eegDataProcessor.trials,
                                                      som,
                                                      response_PSIs_for_all_colors_matrix_array,
                                                      pathLeft,
                                                      params,
                                                      no_samples,
                                                      alignment=Alignment.LEFT)

        PlotsGenerator.groupByResponseWithPsiUsingBMU(eegDataProcessor.trials,
                                                      som,
                                                      response_PSIs_for_all_colors_matrix_array,
                                                      pathRight,
                                                      params,
                                                      no_samples,
                                                      alignment=Alignment.RIGHT)

        # -------------------------------------------------------------------

        PlotsGenerator.groupByStimulusWithPsiUsingBMU(eegDataProcessor.trials,
                                                      som,
                                                      stimulus_PSIs_for_all_colors_matrix_array,
                                                      pathLeft,
                                                      params,
                                                      no_samples,
                                                      alignment=Alignment.LEFT)

        PlotsGenerator.groupByStimulusWithPsiUsingBMU(eegDataProcessor.trials,
                                                      som,
                                                      stimulus_PSIs_for_all_colors_matrix_array,
                                                      pathRight,
                                                      params,
                                                      no_samples,
                                                      alignment=Alignment.RIGHT)
        # -------------------------------------------------------------------

        PlotsGenerator.groupByVisibilityWithPsiUsingBMU(eegDataProcessor.trials,
                                                        som,
                                                        visibility_PSIs_for_all_colors_matrix_array,
                                                        pathLeft,
                                                        params,
                                                        no_samples,
                                                        alignment=Alignment.LEFT)

        PlotsGenerator.groupByVisibilityWithPsiUsingBMU(eegDataProcessor.trials,
                                                        som,
                                                        visibility_PSIs_for_all_colors_matrix_array,
                                                        pathRight,
                                                        params,
                                                        no_samples,
                                                        alignment=Alignment.RIGHT)