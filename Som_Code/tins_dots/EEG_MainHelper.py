from tins_dots.Plots_Generator import Alignment, PlotsGenerator


class EEG_MainHelper:

    @staticmethod
    def main_with_psi1(list_freq_by_response, list_freq_by_stimulus, list_freq_by_visibility, som, eegDataProcessor, pathLeft, pathRight, params, no_samples, weighted=False):
        response_PSIs_for_all_colors_matrix_array = PlotsGenerator.computePSIByGroup(list_freq_by_response,
                                                                                     som)
        stimulus_PSIs_for_all_colors_matrix_array = PlotsGenerator.computePSIByGroup(list_freq_by_stimulus,
                                                                                     som)

        visibility_PSIs_for_all_colors_matrix_array = PlotsGenerator.computePSIByGroup(list_freq_by_visibility,
                                                                                       som)

        print("Response PSIs: ")
        print(response_PSIs_for_all_colors_matrix_array)
        print("-------------------------------------------------------------------")
        print("Stimulus PSIs: ")
        print(stimulus_PSIs_for_all_colors_matrix_array)
        print("-------------------------------------------------------------------------")
        print("Visibility PSIs: ")
        print(visibility_PSIs_for_all_colors_matrix_array)
        print("-----------------------------------------------------------------------------")

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

        print("Response PSIs: ")
        print(response_PSIs_for_all_colors_matrix_array)
        print("-------------------------------------------------------------------")
        print("Stimulus PSIs: ")
        print(stimulus_PSIs_for_all_colors_matrix_array)
        print("-------------------------------------------------------------------------")
        print("Visibility PSIs: ")
        print(visibility_PSIs_for_all_colors_matrix_array)
        print("-----------------------------------------------------------------------------")

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