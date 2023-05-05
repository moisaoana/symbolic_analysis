import numpy as np

from tins_dots.EEG_DataProcessor import EEG_Trial
from tins_dots.Plots_Generator import Alignment, PlotsGenerator, Method


class EEG_MainHelper:

    @staticmethod
    def take_minimum_window_from_trials_start(all_trials, trials_lengths):
        new_all_trials = []
        minimum_trial_length = min(trials_lengths)
        for cnt, trial in enumerate(all_trials):
            new_all_trials.append(
                EEG_Trial(
                    start_timestamp=0,
                    end_timestamp=0,
                    trial_data = trial.trial_data[0:minimum_trial_length]
                )
            )
            #all_trials[cnt].trial_data = trial.trial_data[0:minimum_trial_length]
        return new_all_trials

    @staticmethod
    def take_minimum_window_from_trials_end(all_trials, trials_lengths):
        new_all_trials = []
        minimum_trial_length = min(trials_lengths)
        for cnt, trial in enumerate(all_trials):
            trial_len = len(all_trials[cnt].trial_data)
            #all_trials[cnt].trial_data = trial.trial_data[trial_len - minimum_trial_length:trial_len]
            new_all_trials.append(
                EEG_Trial(
                    start_timestamp=0,
                    end_timestamp=0,
                    trial_data = trial.trial_data[trial_len - minimum_trial_length:trial_len]
                )
            )
        return new_all_trials

    @staticmethod
    def main_with_psi1(list_freq_by_response, list_freq_by_stimulus, list_freq_by_visibility, som, eegDataProcessorTrials,
                       pathLeft, pathRight, pathWindow, params, no_samples, coeff, weighted=False, window=False):
        response_PSIs_for_all_colors_matrix_array = PlotsGenerator.computePSIByGroup(list_freq_by_response,
                                                                                     som)

        """
        stimulus_PSIs_for_all_colors_matrix_array = PlotsGenerator.computePSIByGroup(list_freq_by_stimulus,
                                                                                     som)

        visibility_PSIs_for_all_colors_matrix_array = PlotsGenerator.computePSIByGroup(list_freq_by_visibility,
                                                                                      som)
        """

        if not window:

            PlotsGenerator.groupByResponseWithPsiUsingBMU(eegDataProcessorTrials,
                                                          som,
                                                          response_PSIs_for_all_colors_matrix_array,
                                                          pathLeft,
                                                          params,
                                                          no_samples,
                                                          coeff,
                                                          alignment=Alignment.LEFT,
                                                          weighted=weighted,
                                                          psi_version=1)

            PlotsGenerator.groupByResponseWithPsiUsingBMU(eegDataProcessorTrials,
                                                          som,
                                                          response_PSIs_for_all_colors_matrix_array,
                                                          pathRight,
                                                          params,
                                                          no_samples,
                                                          coeff,
                                                          alignment=Alignment.RIGHT,
                                                          weighted=weighted,
                                                          psi_version=1)

            # -------------------------------------------------------------------
            """"
            PlotsGenerator.groupByStimulusWithPsiUsingBMU(eegDataProcessorTrials,
                                                          som,
                                                          stimulus_PSIs_for_all_colors_matrix_array,
                                                          pathLeft,
                                                          params,
                                                          no_samples,
                                                          coeff,
                                                          alignment=Alignment.LEFT,
                                                          weighted=weighted,
                                                          psi_version=1)

            PlotsGenerator.groupByStimulusWithPsiUsingBMU(eegDataProcessorTrials,
                                                          som,
                                                          stimulus_PSIs_for_all_colors_matrix_array,
                                                          pathRight,
                                                          params,
                                                          no_samples,
                                                          coeff,
                                                          alignment=Alignment.RIGHT,
                                                          weighted=weighted,
                                                          psi_version=1)
            # -------------------------------------------------------------------

            PlotsGenerator.groupByVisibilityWithPsiUsingBMU(eegDataProcessorTrials,
                                                            som,
                                                            visibility_PSIs_for_all_colors_matrix_array,
                                                            pathLeft,
                                                            params,
                                                            no_samples,
                                                            coeff,
                                                            alignment=Alignment.LEFT,
                                                            weighted=weighted,
                                                            psi_version=1)

            PlotsGenerator.groupByVisibilityWithPsiUsingBMU(eegDataProcessorTrials,
                                                            som,
                                                            visibility_PSIs_for_all_colors_matrix_array,
                                                            pathRight,
                                                            params,
                                                            no_samples,
                                                            coeff,
                                                            alignment=Alignment.RIGHT,
                                                            weighted=weighted,
                                                            psi_version=1)
            """
        else:
            PlotsGenerator.groupByResponseWithPsiUsingBMU(eegDataProcessorTrials,
                                                          som,
                                                          response_PSIs_for_all_colors_matrix_array,
                                                          pathWindow,
                                                          params,
                                                          no_samples,
                                                          coeff,
                                                          alignment=Alignment.LEFT,
                                                          weighted=weighted,
                                                          psi_version=1)
            # -------------------------------------------------------------------
            """"
            PlotsGenerator.groupByStimulusWithPsiUsingBMU(eegDataProcessorTrials,
                                                          som,
                                                          stimulus_PSIs_for_all_colors_matrix_array,
                                                          pathWindow,
                                                          params,
                                                          no_samples,
                                                          coeff,
                                                          alignment=Alignment.LEFT,
                                                          weighted=weighted,
                                                          psi_version=1)

            # -------------------------------------------------------------------

            PlotsGenerator.groupByVisibilityWithPsiUsingBMU(eegDataProcessorTrials,
                                                            som,
                                                            visibility_PSIs_for_all_colors_matrix_array,
                                                            pathWindow,
                                                            params,
                                                            no_samples,
                                                            coeff,
                                                            alignment=Alignment.LEFT,
                                                            weighted=weighted,
                                                            psi_version=1)
            """

    @staticmethod
    def main_with_psi2(list_freq_by_response, list_freq_by_stimulus, list_freq_by_visibility, som, eegDataProcessorTrials,
                       pathLeft, pathRight, pathWindow, params, no_samples, coeff, window=False):
        response_PSIs_for_all_colors_matrix_array = PlotsGenerator.computePSIByGroupMethod2(list_freq_by_response,
                                                                                            som)
        stimulus_PSIs_for_all_colors_matrix_array = PlotsGenerator.computePSIByGroupMethod2(list_freq_by_stimulus,
                                                                                            som)

        visibility_PSIs_for_all_colors_matrix_array = PlotsGenerator.computePSIByGroupMethod2(list_freq_by_visibility,
                                                                                              som)
        if not window:
            PlotsGenerator.groupByResponseWithPsiUsingBMU(eegDataProcessorTrials,
                                                          som,
                                                          response_PSIs_for_all_colors_matrix_array,
                                                          pathLeft,
                                                          params,
                                                          no_samples,
                                                          coeff,
                                                          alignment=Alignment.LEFT,
                                                          psi_version=2)

            PlotsGenerator.groupByResponseWithPsiUsingBMU(eegDataProcessorTrials,
                                                          som,
                                                          response_PSIs_for_all_colors_matrix_array,
                                                          pathRight,
                                                          params,
                                                          no_samples,
                                                          coeff,
                                                          alignment=Alignment.RIGHT,
                                                          psi_version=2)

            # -------------------------------------------------------------------

            PlotsGenerator.groupByStimulusWithPsiUsingBMU(eegDataProcessorTrials,
                                                          som,
                                                          stimulus_PSIs_for_all_colors_matrix_array,
                                                          pathLeft,
                                                          params,
                                                          no_samples,
                                                          coeff,
                                                          alignment=Alignment.LEFT,
                                                          psi_version=2)

            PlotsGenerator.groupByStimulusWithPsiUsingBMU(eegDataProcessorTrials,
                                                          som,
                                                          stimulus_PSIs_for_all_colors_matrix_array,
                                                          pathRight,
                                                          params,
                                                          no_samples,
                                                          coeff,
                                                          alignment=Alignment.RIGHT,
                                                          psi_version=2)
            # -------------------------------------------------------------------

            PlotsGenerator.groupByVisibilityWithPsiUsingBMU(eegDataProcessorTrials,
                                                            som,
                                                            visibility_PSIs_for_all_colors_matrix_array,
                                                            pathLeft,
                                                            params,
                                                            no_samples,
                                                            coeff,
                                                            alignment=Alignment.LEFT,
                                                            psi_version=2)

            PlotsGenerator.groupByVisibilityWithPsiUsingBMU(eegDataProcessorTrials,
                                                            som,
                                                            visibility_PSIs_for_all_colors_matrix_array,
                                                            pathRight,
                                                            params,
                                                            no_samples,
                                                            coeff,
                                                            alignment=Alignment.RIGHT,
                                                            psi_version=2)
        else:
            PlotsGenerator.groupByResponseWithPsiUsingBMU(eegDataProcessorTrials,
                                                          som,
                                                          response_PSIs_for_all_colors_matrix_array,
                                                          pathWindow,
                                                          params,
                                                          no_samples,
                                                          coeff,
                                                          alignment=Alignment.LEFT,
                                                          psi_version=2)

            # -------------------------------------------------------------------

            PlotsGenerator.groupByStimulusWithPsiUsingBMU(eegDataProcessorTrials,
                                                          som,
                                                          stimulus_PSIs_for_all_colors_matrix_array,
                                                          pathWindow,
                                                          params,
                                                          no_samples,
                                                          coeff,
                                                          alignment=Alignment.LEFT,
                                                          psi_version=2)
            # -------------------------------------------------------------------

            PlotsGenerator.groupByVisibilityWithPsiUsingBMU(eegDataProcessorTrials,
                                                            som,
                                                            visibility_PSIs_for_all_colors_matrix_array,
                                                            pathWindow,
                                                            params,
                                                            no_samples,
                                                            coeff,
                                                            alignment=Alignment.LEFT,
                                                            psi_version=2)

    @staticmethod
    def full_pipeline(eegDataProcessor, som, pathLeft, pathRight, pathWindowStart, pathWindowEnd, params, coeff, no_samples):

        # LEFT, RIGHT -----------------------------------------------
        # lista de liste (lista pt nothing, lista pt smth, lista pt identified), fiecare lista contine freq matrix pt each trial

        list_freq_by_response = PlotsGenerator.groupByResponseV2(eegDataProcessor.trials, som, pathLeft, params, alignment=Alignment.LEFT, method=Method.BMU)
        PlotsGenerator.groupByResponseV2(eegDataProcessor.trials, som, pathRight, params, alignment=Alignment.RIGHT, method=Method.BMU)

        """
        list_freq_by_stimulus = PlotsGenerator.groupByStimulusV2(eegDataProcessor.trials, som, pathLeft, params, alignment=Alignment.LEFT, method=Method.BMU)
        PlotsGenerator.groupByStimulusV2(eegDataProcessor.trials, som, pathRight, params, alignment=Alignment.RIGHT, method=Method.BMU)

        list_freq_by_visibility = PlotsGenerator.groupByVisibilityV2(eegDataProcessor.trials, som, pathLeft, params, alignment=Alignment.LEFT, method=Method.BMU)
        PlotsGenerator.groupByVisibilityV2(eegDataProcessor.trials, som, pathRight, params, alignment=Alignment.RIGHT, method=Method.BMU)
        """
        EEG_MainHelper.main_with_psi1(list_freq_by_response, [], [],
                                      som, eegDataProcessor.trials, pathLeft, pathRight, '', params,
                                      no_samples, coeff, weighted=True, window=False)
        """"
        EEG_MainHelper.main_with_psi2(list_freq_by_response, list_freq_by_stimulus, list_freq_by_visibility, som,
                                      eegDataProcessor.trials, pathLeft, pathRight, '', params, no_samples, coeff,
                                      window=False)
        """

        # WINDOW-----------------
        window_start_trials = EEG_MainHelper.take_minimum_window_from_trials_start(eegDataProcessor.trials, eegDataProcessor.trials_lengths)
        list_freq_by_response = PlotsGenerator.groupByResponseV2( window_start_trials, som, pathWindowStart,params, alignment=Alignment.LEFT, method=Method.BMU)
        #list_freq_by_stimulus = PlotsGenerator.groupByStimulusV2( window_start_trials, som, pathWindowStart, params, alignment=Alignment.LEFT, method=Method.BMU)
        #list_freq_by_visibility = PlotsGenerator.groupByVisibilityV2( window_start_trials, som, pathWindowStart, params, alignment=Alignment.LEFT, method=Method.BMU)
        EEG_MainHelper.main_with_psi1(list_freq_by_response, [], [],
                                      som,  window_start_trials, '', '', pathWindowStart, params,
                                      no_samples, coeff, weighted=True, window=True)
        """
        EEG_MainHelper.main_with_psi2(list_freq_by_response, [], [], som,
                                       window_start_trials, '', '', pathWindowStart, params, no_samples, coeff,
                                      window=True)
        """

        window_end_trials = EEG_MainHelper.take_minimum_window_from_trials_end(eegDataProcessor.trials, eegDataProcessor.trials_lengths)
        list_freq_by_response = PlotsGenerator.groupByResponseV2(window_end_trials, som, pathWindowEnd, params,
                                                                 alignment=Alignment.LEFT, method=Method.BMU)
        #list_freq_by_stimulus = PlotsGenerator.groupByStimulusV2(window_end_trials, som, pathWindowEnd, params,
        #                                                         alignment=Alignment.LEFT, method=Method.BMU)
        #list_freq_by_visibility = PlotsGenerator.groupByVisibilityV2(window_end_trials, som, pathWindowEnd,
        #                                                             params, alignment=Alignment.LEFT,
        #                                                             method=Method.BMU)
        EEG_MainHelper.main_with_psi1(list_freq_by_response, [], [],
                                      som, window_end_trials, '', '', pathWindowEnd, params,
                                      no_samples, coeff, weighted=True, window=True)
        """
        EEG_MainHelper.main_with_psi2(list_freq_by_response, list_freq_by_stimulus, list_freq_by_visibility, som,
                                      window_end_trials, '', '', pathWindowEnd, params, no_samples, coeff, window=True)
        """

