import io
import math

import matplotlib
import numpy as np
import plotly.graph_objs as go
from mpl_toolkits.mplot3d import Axes3D
from plotly.subplots import make_subplots
import plotly.io as pio
from matplotlib import pyplot as plt
from reportlab.lib.pagesizes import A4
from mayavi import mlab

from PIL import Image

from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from enum import Enum

# matplotlib.use('Qt5Agg')
from Som_Code.readerUtils import ReaderUtils
from Som_Code.utils import Utils
matplotlib.use("Agg")


class GroupingMethod(Enum):
    BY_RESPONSE = 1
    BY_VISIBILITY = 2
    BY_STIMULUS = 3


class Alignment(Enum):
    LEFT = 1
    RIGHT = 2


class Method(Enum):
    BMU = 1
    CLUSTERS = 2




class PlotsGenerator:

    # get data functions -----------------------------------------------------------------------------

    @staticmethod
    def getNothingData(figure_array):
        return [*figure_array[0:61], *figure_array[62:69], *figure_array[70:76], *figure_array[77:90],
                *figure_array[95:96], *figure_array[104:105], *figure_array[107:108], *figure_array[111:117],
                *figure_array[120:121]]

    @staticmethod
    def getSomethingData(figure_array):
        return [*figure_array[61:62], *figure_array[69:70], *figure_array[76:77], *figure_array[90:95],
                *figure_array[96:104], *figure_array[105:107], *figure_array[108:111], *figure_array[112:114],
                *figure_array[118:120], *figure_array[123:124], *figure_array[134:138], *figure_array[139:141],
                *figure_array[142:143], *figure_array[144:152], *figure_array[156:158], *figure_array[162:163],
                *figure_array[164:166], *figure_array[169:170], *figure_array[174:175], *figure_array[177:179],
                *figure_array[180:183]]

    @staticmethod
    def getIdentifiedData(figure_array):
        return [*figure_array[117:118], *figure_array[121:123], *figure_array[124:132], *figure_array[133:134],
                *figure_array[138:139], *figure_array[141:142], *figure_array[143:144], *figure_array[152:156],
                *figure_array[158:162], *figure_array[163:164], *figure_array[166:169], *figure_array[170:174],
                *figure_array[175:177], *figure_array[179:180], *figure_array[183:210], *figure_array[132:133]]

    @staticmethod
    def getStimulusPosetaData(figure_array):
        return [figure_array[0], figure_array[31], figure_array[82], figure_array[106], figure_array[144],
                figure_array[157], figure_array[185]]

    @staticmethod
    def getStimulusToporData(figure_array):
        return [figure_array[1], figure_array[49], figure_array[69], figure_array[97], figure_array[143],
                figure_array[170], figure_array[190]]

    @staticmethod
    def getStimulusOalaData(figure_array):
        return [figure_array[2], figure_array[54], figure_array[85], figure_array[119], figure_array[146],
                figure_array[159], figure_array[199]]

    @staticmethod
    def getStimulusElicopterData(figure_array):
        return [figure_array[3], figure_array[41], figure_array[77], figure_array[99], figure_array[133],
                figure_array[171], figure_array[192]]

    @staticmethod
    def getStimulusUrsData(figure_array):
        return [figure_array[4], figure_array[36], figure_array[86], figure_array[93], figure_array[123],
                figure_array[169], figure_array[181]]

    @staticmethod
    def getStimulusPalarieData(figure_array):
        return [figure_array[5], figure_array[43], figure_array[83], figure_array[100], figure_array[126],
                figure_array[154], figure_array[194]]

    @staticmethod
    def getStimulusFoarfeceData(figure_array):
        return [figure_array[6], figure_array[59], figure_array[78], figure_array[118], figure_array[142],
                figure_array[165], figure_array[198]]

    @staticmethod
    def getStimulusBananaData(figure_array):
        return [figure_array[7], figure_array[51], figure_array[70], figure_array[110], figure_array[149],
                figure_array[155], figure_array[193]]

    @staticmethod
    def getStimulusLampaData(figure_array):
        return [figure_array[8], figure_array[30], figure_array[74], figure_array[114], figure_array[136],
                figure_array[156], figure_array[182]]

    @staticmethod
    def getStimulusChitaraData(figure_array):
        return [figure_array[9], figure_array[34], figure_array[76], figure_array[113], figure_array[138],
                figure_array[166], figure_array[195]]

    @staticmethod
    def getStimulusMasinaData(figure_array):
        return [figure_array[10], figure_array[56], figure_array[61], figure_array[102], figure_array[145],
                figure_array[168], figure_array[203]]

    @staticmethod
    def getStimulusVacaData(figure_array):
        return [figure_array[11], figure_array[45], figure_array[63], figure_array[94], figure_array[129],
                figure_array[172], figure_array[206]]

    @staticmethod
    def getStimulusFurculitaData(figure_array):
        return [figure_array[12], figure_array[47], figure_array[73], figure_array[90], figure_array[127],
                figure_array[152], figure_array[205]]

    @staticmethod
    def getStimulusCerbData(figure_array):
        return [figure_array[13], figure_array[48], figure_array[65], figure_array[117], figure_array[131],
                figure_array[163], figure_array[189]]

    @staticmethod
    def getStimulusPantaloniData(figure_array):
        return [figure_array[14], figure_array[35], figure_array[79], figure_array[96], figure_array[134],
                figure_array[176], figure_array[184]]

    @staticmethod
    def getStimulusScaunData(figure_array):
        return [figure_array[15], figure_array[52], figure_array[80], figure_array[112], figure_array[130],
                figure_array[160], figure_array[197]]

    @staticmethod
    def getStimulusPesteData(figure_array):
        return [figure_array[16], figure_array[37], figure_array[89], figure_array[105], figure_array[122],
                figure_array[161], figure_array[187]]

    @staticmethod
    def getStimulusCaineData(figure_array):
        return [figure_array[17], figure_array[32], figure_array[67], figure_array[91], figure_array[128],
                figure_array[167], figure_array[191]]

    @staticmethod
    def getStimulusSticlaData(figure_array):
        return [figure_array[18], figure_array[53], figure_array[68], figure_array[95], figure_array[120],
                figure_array[178], figure_array[204]]

    @staticmethod
    def getStimulusPistolData(figure_array):
        return [figure_array[19], figure_array[58], figure_array[64], figure_array[101], figure_array[132],
                figure_array[173], figure_array[200]]

    @staticmethod
    def getStimulusBicicletaData(figure_array):
        return [figure_array[20], figure_array[39], figure_array[81], figure_array[115], figure_array[140],
                figure_array[162], figure_array[196]]

    @staticmethod
    def getStimulusCalData(figure_array):
        return [figure_array[21], figure_array[44], figure_array[75], figure_array[98], figure_array[139],
                figure_array[179], figure_array[183]]

    @staticmethod
    def getStimulusElefantData(figure_array):
        return [figure_array[22], figure_array[43], figure_array[88], figure_array[107], figure_array[148],
                figure_array[174], figure_array[201]]

    @staticmethod
    def getStimulusIepureData(figure_array):
        return [figure_array[23], figure_array[55], figure_array[60], figure_array[103], figure_array[141],
                figure_array[151], figure_array[188]]

    @staticmethod
    def getStimulusPaharData(figure_array):
        return [figure_array[24], figure_array[46], figure_array[72], figure_array[111], figure_array[137],
                figure_array[164], figure_array[180]]

    @staticmethod
    def getStimulusMasaData(figure_array):
        return [figure_array[25], figure_array[38], figure_array[87], figure_array[109], figure_array[124],
                figure_array[175], figure_array[207]]

    @staticmethod
    def getStimulusUmbrelaData(figure_array):
        return [figure_array[26], figure_array[50], figure_array[66], figure_array[108], figure_array[125],
                figure_array[150], figure_array[209]]

    @staticmethod
    def getStimulusFlutureData(figure_array):
        return [figure_array[27], figure_array[33], figure_array[62], figure_array[118], figure_array[121],
                figure_array[153], figure_array[186]]

    @staticmethod
    def getStimulusGirafaData(figure_array):
        return [figure_array[28], figure_array[57], figure_array[71], figure_array[92], figure_array[135],
                figure_array[158], figure_array[208]]

    @staticmethod
    def getStimulusPianData(figure_array):
        return [figure_array[29], figure_array[40], figure_array[84], figure_array[104], figure_array[147],
                figure_array[177], figure_array[202]]

    # PSI plots ------------------------------------------------------------------------------------------------

    @staticmethod
    def getNewFigureArrayUsingPSI(som, trials, PSI_matrix, mean, st_dev, coeff, ssd=False, alignment=Alignment.LEFT):
        barcodes_array = []
        all_color_arrays = []
        max_length_color_array = 0
        for cnt, trial in enumerate(trials):
            if cnt % 1000 == 0:
                print("Trial PSI " + str(cnt))
            if ssd:
                colors_array, _ = Utils.get_colors_array(trial, som)
            else:
                colors_array, _ = Utils.get_colors_array(trial.trial_data, som)
            for i, color in enumerate(colors_array):
                color_PSI = PSI_matrix[int(color[0] * som.getX())][int(color[1] * som.getY())][
                    int(color[2] * som.getZ())]
                if color_PSI-mean < coeff*st_dev:
                    colors_array[i] = [1, 1, 1]
            all_color_arrays.append(colors_array)
            if len(colors_array) > max_length_color_array:
                max_length_color_array = len(colors_array)
        for cnt, colors_array in enumerate(all_color_arrays):
            if len(colors_array) != max_length_color_array:
                for i in range(0, max_length_color_array - len(colors_array)):
                    if alignment == Alignment.LEFT:
                        colors_array.append([1, 1, 1])
                    else:
                        colors_array.insert(0, [1, 1, 1])
            figure_data_tuple = PlotsGenerator.generateColorSequenceForTrialMatplotlib(colors_array)
            barcodes_array.append(figure_data_tuple)
        return barcodes_array

    @staticmethod
    def computeWeightedPSI(list_trials, list_psi, number_of_samples):
        total_samples_category = []
        for list_element in list_trials:
            total_samples_category.append(0)

        for cnt, each_list_trials in enumerate(list_trials):
            for trial in each_list_trials:
                total_samples_category[cnt] += len(trial.trial_data)

        list_probabilities = []

        for category in total_samples_category:
            list_probabilities.append(category / number_of_samples)

        weights_each_category_list = []

        first_weight_denominator = 1
        for cnt, probability in enumerate(list_probabilities):
            if cnt != 0:
                first_weight_denominator += list_probabilities[0] / probability

        weights_each_category_list.append(1 / first_weight_denominator)

        for cnt, probability in enumerate(list_probabilities):
            if cnt != 0:
                weights_each_category_list.append((list_probabilities[0] / probability) * weights_each_category_list[0])

        for cnt, _ in enumerate(list_psi):
            list_psi[cnt] *= weights_each_category_list[cnt]

        return list_psi

    @staticmethod
    def findThresholdForGroupBasedOnPsiArray(psi_array, som):

        sum_avg = 0
        total_size = len(psi_array) * som.getX() * som.getY() * som.getZ()

        for psi_matrix in psi_array:
            for x in range(0, som.getX()):
                for y in range(0, som.getY()):
                    for z in range(0, som.getZ()):
                        sum_avg += psi_matrix[x][y][z]

        return 1.1 * (sum_avg / total_size)

    @staticmethod
    def computeMeanAndStDevForGroup(psi_array, som):
        #with mean and st dev

        sum_avg = 0
        total_size = len(psi_array) * som.getX() * som.getY() * som.getZ()

        for psi_matrix in psi_array:
            for x in range(0, som.getX()):
                for y in range(0, som.getY()):
                    for z in range(0, som.getZ()):
                        sum_avg += psi_matrix[x][y][z]

        mean = sum_avg / total_size
        st_dev = 0

        for psi_matrix in psi_array:
            for x in range(0, som.getX()):
                for y in range(0, som.getY()):
                    for z in range(0, som.getZ()):
                        st_dev += (psi_matrix[x][y][z]-mean)**2

        st_dev = math.sqrt(st_dev/total_size)
        return mean, st_dev



    @staticmethod
    def groupByResponseWithPsiUsingBMU(all_trials_data, som, psi_array, path, params, number_of_samples, coeff,
                                       ssd=False, alignment=Alignment.LEFT, weighted=False, psi_version=1):
        nothing_trials = PlotsGenerator.getNothingData(all_trials_data)
        something_trials = PlotsGenerator.getSomethingData(all_trials_data)
        identified_trials = PlotsGenerator.getIdentifiedData(all_trials_data)

        nothing_trials = PlotsGenerator.sortTrials(nothing_trials)
        something_trials = PlotsGenerator.sortTrials(something_trials)
        identified_trials = PlotsGenerator.sortTrials(identified_trials)

        copy_psi_array = psi_array.copy()

        if weighted:
            list_trials_by_group = [nothing_trials, something_trials, identified_trials]
            copy_psi_array = PlotsGenerator.computeWeightedPSI(list_trials_by_group, copy_psi_array, number_of_samples)

        #threshold = PlotsGenerator.findThresholdForGroupBasedOnPsiArray(copy_psi_array, som)
        mean, st_dev = PlotsGenerator.computeMeanAndStDevForGroup(copy_psi_array, som)

        nothing_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, nothing_trials, copy_psi_array[0], mean, st_dev, coeff,
                                                                   alignment=alignment)
        something_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, something_trials, copy_psi_array[1],
                                                                     mean, st_dev, coeff,
                                                                     alignment=alignment)
        identified_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, identified_trials, copy_psi_array[2],
                                                                      mean, st_dev, coeff,
                                                                      alignment=alignment)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(nothing_figures, n_rows=len(nothing_figures), n_cols=1)
        plt.suptitle("Response: nothing, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "response_nothing_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(something_figures, n_rows=len(something_figures),
                                                                n_cols=1)
        plt.suptitle("PSI - Response: something, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "response_something_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(identified_figures, n_rows=len(identified_figures),
                                                                n_cols=1)
        plt.suptitle(
            "PSI - Response: what the subject sees (correct + 1 incorrect), PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "response_identified_psi"+str(psi_version)+".png", dpi=300)
        # plt.show()

    @staticmethod
    def groupByStimulusWithPsiUsingBMU(all_trials_data, som, psi_array, path, params, number_of_samples, coeff,
                                       ssd=False, alignment=Alignment.LEFT, weighted=False, psi_version=1):

        poseta_trials = PlotsGenerator.getStimulusPosetaData(all_trials_data)
        topor_trials = PlotsGenerator.getStimulusToporData(all_trials_data)
        oala_trials = PlotsGenerator.getStimulusOalaData(all_trials_data)
        elicopter_trials = PlotsGenerator.getStimulusElicopterData(all_trials_data)
        urs_trials = PlotsGenerator.getStimulusUrsData(all_trials_data)
        palarie_trials = PlotsGenerator.getStimulusPalarieData(all_trials_data)
        foarfece_trials = PlotsGenerator.getStimulusFoarfeceData(all_trials_data)
        banana_trials = PlotsGenerator.getStimulusBananaData(all_trials_data)
        lampa_trials = PlotsGenerator.getStimulusLampaData(all_trials_data)
        chitara_trials = PlotsGenerator.getStimulusChitaraData(all_trials_data)
        masina_trials = PlotsGenerator.getStimulusMasinaData(all_trials_data)
        vaca_trials = PlotsGenerator.getStimulusVacaData(all_trials_data)
        furculita_trials = PlotsGenerator.getStimulusFurculitaData(all_trials_data)
        cerb_trials = PlotsGenerator.getStimulusCerbData(all_trials_data)
        pantaloni_trials = PlotsGenerator.getStimulusPantaloniData(all_trials_data)
        scaun_trials = PlotsGenerator.getStimulusScaunData(all_trials_data)
        peste_trials = PlotsGenerator.getStimulusPesteData(all_trials_data)
        caine_trials = PlotsGenerator.getStimulusCaineData(all_trials_data)
        sticla_trials = PlotsGenerator.getStimulusSticlaData(all_trials_data)
        pistol_trials = PlotsGenerator.getStimulusPistolData(all_trials_data)
        bicicleta_trials = PlotsGenerator.getStimulusBicicletaData(all_trials_data)
        cal_trials = PlotsGenerator.getStimulusCalData(all_trials_data)
        elefant_trials = PlotsGenerator.getStimulusElefantData(all_trials_data)
        iepure_trials = PlotsGenerator.getStimulusIepureData(all_trials_data)
        pahar_trials = PlotsGenerator.getStimulusPaharData(all_trials_data)
        masa_trials = PlotsGenerator.getStimulusMasaData(all_trials_data)
        umbrela_trials = PlotsGenerator.getStimulusUmbrelaData(all_trials_data)
        fluture_trials = PlotsGenerator.getStimulusFlutureData(all_trials_data)
        girafa_trials = PlotsGenerator.getStimulusGirafaData(all_trials_data)
        pian_trials = PlotsGenerator.getStimulusPianData(all_trials_data)

        poseta_trials = PlotsGenerator.sortTrials(poseta_trials)
        topor_trials = PlotsGenerator.sortTrials(topor_trials)
        oala_trials = PlotsGenerator.sortTrials(oala_trials)
        elicopter_trials = PlotsGenerator.sortTrials(elicopter_trials)
        urs_trials = PlotsGenerator.sortTrials(urs_trials)
        palarie_trials = PlotsGenerator.sortTrials(palarie_trials)
        foarfece_trials = PlotsGenerator.sortTrials(foarfece_trials)
        banana_trials = PlotsGenerator.sortTrials(banana_trials)
        lampa_trials = PlotsGenerator.sortTrials(lampa_trials)
        chitara_trials = PlotsGenerator.sortTrials(chitara_trials)
        masina_trials = PlotsGenerator.sortTrials(masina_trials)
        vaca_trials = PlotsGenerator.sortTrials(vaca_trials)
        furculita_trials = PlotsGenerator.sortTrials(furculita_trials)
        cerb_trials = PlotsGenerator.sortTrials(cerb_trials)
        pantaloni_trials = PlotsGenerator.sortTrials(pantaloni_trials)
        scaun_trials = PlotsGenerator.sortTrials(scaun_trials)
        peste_trials = PlotsGenerator.sortTrials(peste_trials)
        caine_trials = PlotsGenerator.sortTrials(caine_trials)
        sticla_trials = PlotsGenerator.sortTrials(sticla_trials)
        pistol_trials = PlotsGenerator.sortTrials(pistol_trials)
        bicicleta_trials = PlotsGenerator.sortTrials(bicicleta_trials)
        cal_trials = PlotsGenerator.sortTrials(cal_trials)
        elefant_trials = PlotsGenerator.sortTrials(elefant_trials)
        iepure_trials = PlotsGenerator.sortTrials(iepure_trials)
        pahar_trials = PlotsGenerator.sortTrials(pahar_trials)
        masa_trials = PlotsGenerator.sortTrials(masa_trials)
        umbrela_trials = PlotsGenerator.sortTrials(umbrela_trials)
        fluture_trials = PlotsGenerator.sortTrials(fluture_trials)
        girafa_trials = PlotsGenerator.sortTrials(girafa_trials)
        pian_trials = PlotsGenerator.sortTrials(pian_trials)

        copy_psi_array = psi_array.copy()

        if weighted:
            list_trials_by_group = [poseta_trials, topor_trials, oala_trials, elicopter_trials, urs_trials,
                                    palarie_trials,
                                    foarfece_trials, banana_trials,
                                    lampa_trials, chitara_trials, masina_trials, vaca_trials, furculita_trials,
                                    cerb_trials,
                                    pantaloni_trials, scaun_trials, peste_trials, caine_trials,
                                    sticla_trials, pistol_trials, bicicleta_trials, cal_trials, elefant_trials,
                                    iepure_trials, pahar_trials, masa_trials, umbrela_trials,
                                    fluture_trials, girafa_trials, pian_trials]
            copy_psi_array = PlotsGenerator.computeWeightedPSI(list_trials_by_group, copy_psi_array, number_of_samples)

        #threshold = PlotsGenerator.findThresholdForGroupBasedOnPsiArray(copy_psi_array, som)
        mean, st_dev = PlotsGenerator.computeMeanAndStDevForGroup(copy_psi_array, som)

        poseta_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, poseta_trials, copy_psi_array[0], mean, st_dev, coeff,
                                                                  alignment=alignment)
        topor_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, topor_trials, copy_psi_array[1], mean, st_dev, coeff,
                                                                 alignment=alignment)
        oala_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, oala_trials, copy_psi_array[2], mean, st_dev, coeff,
                                                                alignment=alignment)
        elicopter_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, elicopter_trials, copy_psi_array[3],
                                                                     mean, st_dev, coeff,
                                                                     alignment=alignment)
        urs_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, urs_trials, copy_psi_array[4], mean, st_dev, coeff,
                                                               alignment=alignment)
        palarie_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, palarie_trials, copy_psi_array[5], mean, st_dev, coeff,
                                                                   alignment=alignment)
        foarfece_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, foarfece_trials, copy_psi_array[6], mean, st_dev, coeff,
                                                                    alignment=alignment)
        banana_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, banana_trials, copy_psi_array[7], mean, st_dev, coeff,
                                                                  alignment=alignment)
        lampa_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, lampa_trials, copy_psi_array[8], mean, st_dev, coeff,
                                                                 alignment=alignment)
        chitara_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, chitara_trials, copy_psi_array[9], mean, st_dev, coeff,
                                                                   alignment=alignment)
        masina_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, masina_trials, copy_psi_array[10], mean, st_dev, coeff,
                                                                  alignment=alignment)
        vaca_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, vaca_trials, copy_psi_array[11], mean, st_dev, coeff,
                                                                alignment=alignment)
        furculita_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, furculita_trials, copy_psi_array[12],
                                                                     mean, st_dev, coeff,
                                                                     alignment=alignment)
        cerb_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, cerb_trials, copy_psi_array[13], mean, st_dev, coeff,
                                                                alignment=alignment)
        pantaloni_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, pantaloni_trials, copy_psi_array[14],
                                                                     mean, st_dev, coeff,
                                                                     alignment=alignment)
        scaun_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, scaun_trials, copy_psi_array[15], mean, st_dev, coeff,
                                                                 alignment=alignment)
        peste_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, peste_trials, copy_psi_array[16], mean, st_dev, coeff,
                                                                 alignment=alignment)
        caine_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, caine_trials, copy_psi_array[17], mean, st_dev, coeff,
                                                                 alignment=alignment)
        sticla_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, sticla_trials, copy_psi_array[18], mean, st_dev, coeff,
                                                                  alignment=alignment)
        pistol_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, pistol_trials, copy_psi_array[19], mean, st_dev, coeff,
                                                                  alignment=alignment)
        bicicleta_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, bicicleta_trials, copy_psi_array[20],
                                                                     mean, st_dev, coeff,
                                                                     alignment=alignment)
        cal_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, cal_trials, copy_psi_array[21], mean, st_dev, coeff,
                                                               alignment=alignment)
        elefant_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, elefant_trials, copy_psi_array[22], mean, st_dev, coeff,
                                                                   alignment=alignment)
        iepure_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, iepure_trials, copy_psi_array[23], mean, st_dev, coeff,
                                                                  alignment=alignment)
        pahar_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, pahar_trials, copy_psi_array[24], mean, st_dev, coeff,
                                                                 alignment=alignment)
        masa_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, masa_trials, copy_psi_array[25], mean, st_dev, coeff,
                                                                alignment=alignment)
        umbrela_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, umbrela_trials, copy_psi_array[26], mean, st_dev, coeff,
                                                                   alignment=alignment)
        fluture_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, fluture_trials, copy_psi_array[27], mean, st_dev, coeff,
                                                                   alignment=alignment)
        girafa_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, girafa_trials, copy_psi_array[28], mean, st_dev, coeff,
                                                                  alignment=alignment)
        pian_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, pian_trials, copy_psi_array[29], mean, st_dev, coeff,
                                                                alignment=alignment)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(poseta_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: poseta/geanta (de dama), PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_poseta_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(topor_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: topor/secure, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_topor_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(oala_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: oala/cratita, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_oala_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(elicopter_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: elicopter, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_elicopter_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(urs_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: urs (polar), PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_urs_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(palarie_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: palarie, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_palarie_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(foarfece_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: foarfece, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_foarfece_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(banana_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: banana, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_banana_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(lampa_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: lampa/veioza, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_lampa_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(chitara_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: chitara (electrica), PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_chitara_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(masina_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: masina, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_masina_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(vaca_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: vaca, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_vaca_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(furculita_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: furculita, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_furculita_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(cerb_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: cerb, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_cerb_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(pantaloni_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: pantaloni (scurti), PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_pantaloni_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(scaun_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: scaun, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_scaun_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(peste_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: peste, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_peste_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(caine_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: caine/catel, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_caine_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(sticla_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: sticla, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_sticla_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(pistol_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: pistol, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_pistol_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(bicicleta_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: bicicleta, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_bicicleta_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(cal_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: cal, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_cal_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(elefant_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: elefant, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_elefant_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(iepure_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: iepure, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_iepure_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(pahar_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: pahar/cupa, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_cupa_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(masa_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: masa, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_masa_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(umbrela_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: umbrela, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_umbrela_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(fluture_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: fluture, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_fluture_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(girafa_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: girafa, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_girafa_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(pian_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: pian, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_pian_psi"+str(psi_version)+".png", dpi=300)
        # plt.show()

    @staticmethod
    def groupByVisibilityWithPsiUsingBMU(all_trials_data, som, psi_array, path, params, number_of_samples, coeff,
                                         ssd=False, alignment=Alignment.LEFT, weighted=False, psi_version=1):
        v0_trials = all_trials_data[0:30]
        v1_trials = all_trials_data[30:60]
        v2_trials = all_trials_data[60:90]
        v3_trials = all_trials_data[90:120]
        v4_trials = all_trials_data[120:150]
        v5_trials = all_trials_data[150:180]
        v6_trials = all_trials_data[180:210]

        v0_trials = PlotsGenerator.sortTrials(v0_trials)
        v1_trials = PlotsGenerator.sortTrials(v1_trials)
        v2_trials = PlotsGenerator.sortTrials(v2_trials)
        v3_trials = PlotsGenerator.sortTrials(v3_trials)
        v4_trials = PlotsGenerator.sortTrials(v4_trials)
        v5_trials = PlotsGenerator.sortTrials(v5_trials)
        v6_trials = PlotsGenerator.sortTrials(v6_trials)

        copy_psi_array = psi_array.copy()

        if weighted:
            list_trials_by_group = [v0_trials, v1_trials, v2_trials, v3_trials, v4_trials, v5_trials, v6_trials]
            copy_psi_array = PlotsGenerator.computeWeightedPSI(list_trials_by_group, copy_psi_array, number_of_samples)

        #threshold = PlotsGenerator.findThresholdForGroupBasedOnPsiArray(copy_psi_array, som)
        mean, st_dev = PlotsGenerator.computeMeanAndStDevForGroup(copy_psi_array, som)

        v0_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, v0_trials, copy_psi_array[0], mean, st_dev, coeff,
                                                              alignment=alignment)
        v1_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, v1_trials, copy_psi_array[1], mean, st_dev, coeff,
                                                              alignment=alignment)
        v2_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, v2_trials, copy_psi_array[2], mean, st_dev, coeff,
                                                              alignment=alignment)
        v3_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, v3_trials, copy_psi_array[3], mean, st_dev, coeff,
                                                              alignment=alignment)
        v4_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, v4_trials, copy_psi_array[4], mean, st_dev, coeff,
                                                              alignment=alignment)
        v5_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, v5_trials, copy_psi_array[5], mean, st_dev, coeff,
                                                              alignment=alignment)
        v6_figures = PlotsGenerator.getNewFigureArrayUsingPSI(som, v6_trials, copy_psi_array[6], mean, st_dev, coeff,
                                                              alignment=alignment)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(v0_figures, n_rows=30, n_cols=1)
        plt.suptitle("Stimulus visibility: 0.00, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "v0.0_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(v1_figures, n_rows=30, n_cols=1)
        plt.suptitle("Stimulus visibility: 0.05, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "v0.05_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(v2_figures, n_rows=30, n_cols=1)
        plt.suptitle("Stimulus visibility: 0.1, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "v0.1_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(v3_figures, n_rows=30, n_cols=1)
        plt.suptitle("Stimulus visibility: 0.15, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "v0.15_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(v4_figures, n_rows=30, n_cols=1)
        plt.suptitle("Stimulus visibility: 0.2, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "v0.2_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(v5_figures, n_rows=30, n_cols=1)
        plt.suptitle("Stimulus visibility: 0.25, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "v0.25_psi"+str(psi_version)+".png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(v6_figures, n_rows=30, n_cols=1)
        plt.suptitle("Stimulus visibility: 0.3, PSI mean = " + str(round(mean, 3)) + ", st_dev =  " + str(round(st_dev, 3)) + "\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "v0.3_psi"+str(psi_version)+".png", dpi=300)
        # plt.show()

    # compute color frequencies ---------------------------------------------------------------------------------------

    @staticmethod
    def getColorFreqForResponse(color_freq_for_each_trial):
        all_color_freq = []
        nothing_color_freq = PlotsGenerator.getNothingData(color_freq_for_each_trial)

        something_color_freq = PlotsGenerator.getSomethingData(color_freq_for_each_trial)

        identified_color_freq = PlotsGenerator.getIdentifiedData(color_freq_for_each_trial)

        all_color_freq.append(nothing_color_freq)
        all_color_freq.append(something_color_freq)
        all_color_freq.append(identified_color_freq)

        return all_color_freq

    @staticmethod
    def getColorFreqForVisibility(color_freq_for_each_trial):
        all_color_freq = []
        v0_color_freq = color_freq_for_each_trial[0:30]
        v1_color_freq = color_freq_for_each_trial[30:60]
        v2_color_freq = color_freq_for_each_trial[60:90]
        v3_color_freq = color_freq_for_each_trial[90:120]
        v4_color_freq = color_freq_for_each_trial[120:150]
        v5_color_freq = color_freq_for_each_trial[150:180]
        v6_color_freq = color_freq_for_each_trial[180:210]

        all_color_freq.append(v0_color_freq)
        all_color_freq.append(v1_color_freq)
        all_color_freq.append(v2_color_freq)
        all_color_freq.append(v3_color_freq)
        all_color_freq.append(v4_color_freq)
        all_color_freq.append(v5_color_freq)
        all_color_freq.append(v6_color_freq)

        return all_color_freq

    @staticmethod
    def getColorFreqForStimulus(color_freq_for_each_trial):
        all_color_freq = []
        poseta_color_freq = PlotsGenerator.getStimulusPosetaData(color_freq_for_each_trial)
        topor_color_freq = PlotsGenerator.getStimulusToporData(color_freq_for_each_trial)
        oala_color_freq = PlotsGenerator.getStimulusOalaData(color_freq_for_each_trial)
        elicopter_color_freq = PlotsGenerator.getStimulusElicopterData(color_freq_for_each_trial)
        urs_color_freq = PlotsGenerator.getStimulusUrsData(color_freq_for_each_trial)
        palarie_color_freq = PlotsGenerator.getStimulusPalarieData(color_freq_for_each_trial)
        foarfece_color_freq = PlotsGenerator.getStimulusFoarfeceData(color_freq_for_each_trial)
        banana_color_freq = PlotsGenerator.getStimulusBananaData(color_freq_for_each_trial)
        lampa_color_freq = PlotsGenerator.getStimulusLampaData(color_freq_for_each_trial)
        chitara_color_freq = PlotsGenerator.getStimulusChitaraData(color_freq_for_each_trial)
        masina_color_freq = PlotsGenerator.getStimulusMasinaData(color_freq_for_each_trial)
        vaca_color_freq = PlotsGenerator.getStimulusVacaData(color_freq_for_each_trial)
        furculita_color_freq = PlotsGenerator.getStimulusFurculitaData(color_freq_for_each_trial)
        cerb_color_freq = PlotsGenerator.getStimulusCerbData(color_freq_for_each_trial)
        pantaloni_color_freq = PlotsGenerator.getStimulusPantaloniData(color_freq_for_each_trial)
        scaun_color_freq = PlotsGenerator.getStimulusScaunData(color_freq_for_each_trial)
        peste_color_freq = PlotsGenerator.getStimulusPesteData(color_freq_for_each_trial)
        caine_color_freq = PlotsGenerator.getStimulusCaineData(color_freq_for_each_trial)
        sticla_color_freq = PlotsGenerator.getStimulusSticlaData(color_freq_for_each_trial)
        pistol_color_freq = PlotsGenerator.getStimulusPistolData(color_freq_for_each_trial)
        bicicleta_color_freq = PlotsGenerator.getStimulusBicicletaData(color_freq_for_each_trial)
        cal_color_freq = PlotsGenerator.getStimulusCalData(color_freq_for_each_trial)
        elefant_color_freq = PlotsGenerator.getStimulusElefantData(color_freq_for_each_trial)
        iepure_color_freq = PlotsGenerator.getStimulusIepureData(color_freq_for_each_trial)
        pahar_color_freq = PlotsGenerator.getStimulusPaharData(color_freq_for_each_trial)
        masa_color_freq = PlotsGenerator.getStimulusMasaData(color_freq_for_each_trial)
        umbrela_color_freq = PlotsGenerator.getStimulusUmbrelaData(color_freq_for_each_trial)
        fluture_color_freq = PlotsGenerator.getStimulusFlutureData(color_freq_for_each_trial)
        girafa_color_freq = PlotsGenerator.getStimulusGirafaData(color_freq_for_each_trial)
        pian_color_freq = PlotsGenerator.getStimulusPianData(color_freq_for_each_trial)

        all_color_freq.append(poseta_color_freq)
        all_color_freq.append(topor_color_freq)
        all_color_freq.append(oala_color_freq)
        all_color_freq.append(elicopter_color_freq)
        all_color_freq.append(urs_color_freq)
        all_color_freq.append(palarie_color_freq)
        all_color_freq.append(foarfece_color_freq)
        all_color_freq.append(banana_color_freq)
        all_color_freq.append(lampa_color_freq)
        all_color_freq.append(chitara_color_freq)
        all_color_freq.append(masina_color_freq)
        all_color_freq.append(vaca_color_freq)
        all_color_freq.append(furculita_color_freq)
        all_color_freq.append(cerb_color_freq)
        all_color_freq.append(pantaloni_color_freq)
        all_color_freq.append(scaun_color_freq)
        all_color_freq.append(peste_color_freq)
        all_color_freq.append(caine_color_freq)
        all_color_freq.append(sticla_color_freq)
        all_color_freq.append(pistol_color_freq)
        all_color_freq.append(bicicleta_color_freq)
        all_color_freq.append(cal_color_freq)
        all_color_freq.append(elefant_color_freq)
        all_color_freq.append(iepure_color_freq)
        all_color_freq.append(pahar_color_freq)
        all_color_freq.append(masa_color_freq)
        all_color_freq.append(umbrela_color_freq)
        all_color_freq.append(fluture_color_freq)
        all_color_freq.append(girafa_color_freq)
        all_color_freq.append(pian_color_freq)

        return all_color_freq

    @staticmethod
    def computeTotalFrequenciesByGroup(color_freq_list_of_matrices):
        size_tuple = color_freq_list_of_matrices[0].shape
        total_freq_matrix = np.zeros(size_tuple)
        for matrix in color_freq_list_of_matrices:
            total_freq_matrix += matrix
        return total_freq_matrix

    # Compute PSI ---------------------------------------------------------------------------------------------------

    @staticmethod
    def computePSI(group_total_freq_matrix, total_freq_matrix):
        return np.divide(group_total_freq_matrix, total_freq_matrix)

    @staticmethod
    def computePSIByGroup(group_color_freq_array, som):

        group_total_freq_matrix_array = []
        group_PSIs_for_all_colors_matrix_array = []
        """"
        if group == GroupingMethod.BY_RESPONSE:
            group_color_freq_array = PlotsGenerator.getColorFreqForResponse(color_freq_for_each_trial)
        elif group == GroupingMethod.BY_VISIBILITY:
            group_color_freq_array = PlotsGenerator.getColorFreqForVisibility(color_freq_for_each_trial)
        else:
            group_color_freq_array = PlotsGenerator.getColorFreqForStimulus(color_freq_for_each_trial)
        """

        for matrix in group_color_freq_array:
            group_total_freq_matrix_array.append(PlotsGenerator.computeTotalFrequenciesByGroup(matrix))

        size_tuple = (som.getX(), som.getY(), som.getZ())
        total_freq_matrix = np.zeros(size_tuple)

        for matrix in group_total_freq_matrix_array:
            total_freq_matrix += matrix

        for matrix in group_total_freq_matrix_array:
            group_PSIs_for_all_colors_matrix_array.append(PlotsGenerator.computePSI(matrix, total_freq_matrix))

        return group_PSIs_for_all_colors_matrix_array

    @staticmethod
    def computePSIByGroupMethod2(group_color_freq_array, som):

        group_total_freq_matrix_array = []
        group_PSIs_for_all_colors_matrix_array = []
        """"
        if group == GroupingMethod.BY_RESPONSE:
            group_color_freq_array = PlotsGenerator.getColorFreqForResponse(color_freq_for_each_trial)
        elif group == GroupingMethod.BY_VISIBILITY:
            group_color_freq_array = PlotsGenerator.getColorFreqForVisibility(color_freq_for_each_trial)
        else:
            group_color_freq_array = PlotsGenerator.getColorFreqForStimulus(color_freq_for_each_trial)
        """

        for matrix_list in group_color_freq_array:
            group_total_freq_matrix_array.append(PlotsGenerator.computeTotalFrequenciesByGroup(matrix_list))

        total_samples_group = []
        for matrix in group_total_freq_matrix_array:
            total_samples_group.append(np.sum(matrix))

        for cnt, matrix in enumerate(group_total_freq_matrix_array):
            group_PSIs_for_all_colors_matrix_array.append(matrix / total_samples_group[cnt])

        return group_PSIs_for_all_colors_matrix_array

    # Show final plots by group without PSI--------------------------------------------------------------------------

    @staticmethod
    def sortTrials(trials):
        n = len(trials)

        for i in range(n):
            # Last i elements are already sorted
            for j in range(n - i - 1):
                # Swap if the element found is greater than the next element
                if len(trials[j].trial_data) > len(trials[j + 1].trial_data):
                    trials[j], trials[j + 1] = trials[j + 1], trials[j]

        return trials

    @staticmethod
    def groupByResponseV2(all_trials_data, som, path, params, ssd=False, alignment=Alignment.LEFT, method=Method.BMU,
                          markers_and_colors=None, samples_with_clusters=None):

        nothing_trials = PlotsGenerator.getNothingData(all_trials_data)
        something_trials = PlotsGenerator.getSomethingData(all_trials_data)
        identified_trials = PlotsGenerator.getIdentifiedData(all_trials_data)

        # sort after length
        nothing_trials = PlotsGenerator.sortTrials(nothing_trials)
        something_trials = PlotsGenerator.sortTrials(something_trials)
        identified_trials = PlotsGenerator.sortTrials(identified_trials)

        if method == Method.BMU:
            nothing_figures, freq_nothing = PlotsGenerator.getTrialSequencesArrayUsingBMU(nothing_trials, som,
                                                                                          alignment=alignment)
            something_figures, freq_smth = PlotsGenerator.getTrialSequencesArrayUsingBMU(something_trials, som,
                                                                                         alignment=alignment)
            identified_figures, freq_iden = PlotsGenerator.getTrialSequencesArrayUsingBMU(identified_trials, som,
                                                                                          alignment=alignment)
        else:
            nothing_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(nothing_trials, markers_and_colors,
                                                                                 samples_with_clusters,
                                                                                 alignment=alignment)
            something_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(something_trials, markers_and_colors,
                                                                                   samples_with_clusters,
                                                                                   alignment=alignment)
            identified_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(identified_trials,
                                                                                    markers_and_colors,
                                                                                    samples_with_clusters,
                                                                                    alignment=alignment)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(nothing_figures, n_rows=len(nothing_figures),
                                                                n_cols=1)
        plt.suptitle("Response: nothing\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "response_nothing.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(something_figures, n_rows=len(something_figures),
                                                                n_cols=1)
        plt.suptitle("Response: something\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "response_something.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(identified_figures, n_rows=len(identified_figures),
                                                                n_cols=1)
        plt.suptitle("Response: what the subject sees (correct + 1 incorrect)\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "response_identified.png", dpi=300)
        # plt.show()

        if method == Method.BMU:
            all_freq = [freq_nothing, freq_smth, freq_iden]
            return all_freq

    @staticmethod
    def groupByStimulusV2(all_trials_data, som, path, params, ssd=False, alignment=Alignment.LEFT, method=Method.BMU,
                          markers_and_colors=None, samples_with_clusters=None):
        poseta = PlotsGenerator.getStimulusPosetaData(all_trials_data)
        topor = PlotsGenerator.getStimulusToporData(all_trials_data)
        oala = PlotsGenerator.getStimulusOalaData(all_trials_data)
        elicopter = PlotsGenerator.getStimulusElicopterData(all_trials_data)
        urs = PlotsGenerator.getStimulusUrsData(all_trials_data)
        palarie = PlotsGenerator.getStimulusPalarieData(all_trials_data)
        foarfece = PlotsGenerator.getStimulusFoarfeceData(all_trials_data)
        banana = PlotsGenerator.getStimulusBananaData(all_trials_data)
        lampa = PlotsGenerator.getStimulusLampaData(all_trials_data)
        chitara = PlotsGenerator.getStimulusChitaraData(all_trials_data)
        masina = PlotsGenerator.getStimulusMasinaData(all_trials_data)
        vaca = PlotsGenerator.getStimulusVacaData(all_trials_data)
        furculita = PlotsGenerator.getStimulusFurculitaData(all_trials_data)
        cerb = PlotsGenerator.getStimulusCerbData(all_trials_data)
        pantaloni = PlotsGenerator.getStimulusPantaloniData(all_trials_data)
        scaun = PlotsGenerator.getStimulusScaunData(all_trials_data)
        peste = PlotsGenerator.getStimulusPesteData(all_trials_data)
        caine = PlotsGenerator.getStimulusCaineData(all_trials_data)
        sticla = PlotsGenerator.getStimulusSticlaData(all_trials_data)
        pistol = PlotsGenerator.getStimulusPistolData(all_trials_data)
        bicicleta = PlotsGenerator.getStimulusBicicletaData(all_trials_data)
        cal = PlotsGenerator.getStimulusCalData(all_trials_data)
        elefant = PlotsGenerator.getStimulusElefantData(all_trials_data)
        iepure = PlotsGenerator.getStimulusIepureData(all_trials_data)
        pahar = PlotsGenerator.getStimulusPaharData(all_trials_data)
        masa = PlotsGenerator.getStimulusMasaData(all_trials_data)
        umbrela = PlotsGenerator.getStimulusUmbrelaData(all_trials_data)
        fluture = PlotsGenerator.getStimulusFlutureData(all_trials_data)
        girafa = PlotsGenerator.getStimulusGirafaData(all_trials_data)
        pian = PlotsGenerator.getStimulusPianData(all_trials_data)

        # sort after length

        poseta = PlotsGenerator.sortTrials(poseta)
        topor = PlotsGenerator.sortTrials(topor)
        oala = PlotsGenerator.sortTrials(oala)
        elicopter = PlotsGenerator.sortTrials(elicopter)
        urs = PlotsGenerator.sortTrials(urs)
        palarie = PlotsGenerator.sortTrials(palarie)
        foarfece = PlotsGenerator.sortTrials(foarfece)
        banana = PlotsGenerator.sortTrials(banana)
        lampa = PlotsGenerator.sortTrials(lampa)
        chitara = PlotsGenerator.sortTrials(chitara)
        masina = PlotsGenerator.sortTrials(masina)
        vaca = PlotsGenerator.sortTrials(vaca)
        furculita = PlotsGenerator.sortTrials(furculita)
        cerb = PlotsGenerator.sortTrials(cerb)
        pantaloni = PlotsGenerator.sortTrials(pantaloni)
        scaun = PlotsGenerator.sortTrials(scaun)
        peste = PlotsGenerator.sortTrials(peste)
        caine = PlotsGenerator.sortTrials(caine)
        sticla = PlotsGenerator.sortTrials(sticla)
        pistol = PlotsGenerator.sortTrials(pistol)
        bicicleta = PlotsGenerator.sortTrials(bicicleta)
        cal = PlotsGenerator.sortTrials(cal)
        elefant = PlotsGenerator.sortTrials(elefant)
        iepure = PlotsGenerator.sortTrials(iepure)
        pahar = PlotsGenerator.sortTrials(pahar)
        masa = PlotsGenerator.sortTrials(masa)
        umbrela = PlotsGenerator.sortTrials(umbrela)
        fluture = PlotsGenerator.sortTrials(fluture)
        girafa = PlotsGenerator.sortTrials(girafa)
        pian = PlotsGenerator.sortTrials(pian)

        if method == Method.BMU:
            poseta_figures, freq_poseta = PlotsGenerator.getTrialSequencesArrayUsingBMU(poseta, som,
                                                                                        alignment=alignment)
            topor_figures, freq_topor = PlotsGenerator.getTrialSequencesArrayUsingBMU(topor, som,
                                                                                      alignment=alignment)
            oala_figures, freq_oala = PlotsGenerator.getTrialSequencesArrayUsingBMU(oala, som,
                                                                                    alignment=alignment)
            elicopter_figures, freq_elicopter = PlotsGenerator.getTrialSequencesArrayUsingBMU(elicopter, som,
                                                                                              alignment=alignment)
            urs_figures, freq_urs = PlotsGenerator.getTrialSequencesArrayUsingBMU(urs, som,
                                                                                  alignment=alignment)
            palarie_figures, freq_palarie = PlotsGenerator.getTrialSequencesArrayUsingBMU(palarie, som,
                                                                                          alignment=alignment)
            foarfece_figures, freq_foarfece = PlotsGenerator.getTrialSequencesArrayUsingBMU(foarfece, som,
                                                                                            alignment=alignment)
            banana_figures, freq_banana = PlotsGenerator.getTrialSequencesArrayUsingBMU(banana, som,
                                                                                        alignment=alignment)
            lampa_figures, freq_lampa = PlotsGenerator.getTrialSequencesArrayUsingBMU(lampa, som,
                                                                                      alignment=alignment)
            chitara_figures, freq_chitara = PlotsGenerator.getTrialSequencesArrayUsingBMU(chitara, som,
                                                                                          alignment=alignment)
            masina_figures, freq_masina = PlotsGenerator.getTrialSequencesArrayUsingBMU(masina, som,
                                                                                        alignment=alignment)
            vaca_figures, freq_vaca = PlotsGenerator.getTrialSequencesArrayUsingBMU(vaca, som,
                                                                                    alignment=alignment)
            furculita_figures, freq_furculita = PlotsGenerator.getTrialSequencesArrayUsingBMU(furculita, som,
                                                                                              alignment=alignment)
            cerb_figures, freq_cerb = PlotsGenerator.getTrialSequencesArrayUsingBMU(cerb, som,
                                                                                    alignment=alignment)
            pantaloni_figures, freq_pantaloni = PlotsGenerator.getTrialSequencesArrayUsingBMU(pantaloni, som,
                                                                                              alignment=alignment)
            scaun_figures, freq_scaun = PlotsGenerator.getTrialSequencesArrayUsingBMU(scaun, som,
                                                                                      alignment=alignment)
            peste_figures, freq_peste = PlotsGenerator.getTrialSequencesArrayUsingBMU(peste, som,
                                                                                      alignment=alignment)
            caine_figures, freq_caine = PlotsGenerator.getTrialSequencesArrayUsingBMU(caine, som,
                                                                                      alignment=alignment)
            sticla_figures, freq_sticla = PlotsGenerator.getTrialSequencesArrayUsingBMU(sticla, som,
                                                                                        alignment=alignment)
            pistol_figures, freq_pistol = PlotsGenerator.getTrialSequencesArrayUsingBMU(pistol, som,
                                                                                        alignment=alignment)
            bicicleta_figures, freq_bicicleta = PlotsGenerator.getTrialSequencesArrayUsingBMU(bicicleta, som,
                                                                                              alignment=alignment)
            cal_figures, freq_cal = PlotsGenerator.getTrialSequencesArrayUsingBMU(cal, som,
                                                                                  alignment=alignment)
            elefant_figures, freq_elefant = PlotsGenerator.getTrialSequencesArrayUsingBMU(elicopter, som,
                                                                                          alignment=alignment)
            iepure_figures, freq_iepure = PlotsGenerator.getTrialSequencesArrayUsingBMU(iepure, som,
                                                                                        alignment=alignment)
            pahar_figures, freq_pahar = PlotsGenerator.getTrialSequencesArrayUsingBMU(pahar, som,
                                                                                      alignment=alignment)
            masa_figures, freq_masa = PlotsGenerator.getTrialSequencesArrayUsingBMU(masa, som,
                                                                                    alignment=alignment)
            umbrela_figures, freq_umbrela = PlotsGenerator.getTrialSequencesArrayUsingBMU(umbrela, som,
                                                                                          alignment=alignment)
            fluture_figures, freq_fluture = PlotsGenerator.getTrialSequencesArrayUsingBMU(fluture, som,
                                                                                          alignment=alignment)
            girafa_figures, freq_girafa = PlotsGenerator.getTrialSequencesArrayUsingBMU(girafa, som,
                                                                                        alignment=alignment)
            pian_figures, freq_pian = PlotsGenerator.getTrialSequencesArrayUsingBMU(pian, som,
                                                                                    alignment=alignment)
        else:
            poseta_figures= PlotsGenerator.getTrialSequencesArrayUsingClusters(poseta, markers_and_colors, samples_with_clusters, alignment=alignment)
            topor_figures= PlotsGenerator.getTrialSequencesArrayUsingClusters(topor, markers_and_colors, samples_with_clusters, alignment=alignment)
            oala_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(oala, markers_and_colors, samples_with_clusters, alignment=alignment)
            elicopter_figures= PlotsGenerator.getTrialSequencesArrayUsingClusters(elicopter, markers_and_colors, samples_with_clusters, alignment=alignment)
            urs_figures= PlotsGenerator.getTrialSequencesArrayUsingClusters(urs, markers_and_colors, samples_with_clusters, alignment=alignment)
            palarie_figures= PlotsGenerator.getTrialSequencesArrayUsingClusters(palarie, markers_and_colors, samples_with_clusters, alignment=alignment)
            foarfece_figures= PlotsGenerator.getTrialSequencesArrayUsingClusters(foarfece, markers_and_colors, samples_with_clusters, alignment=alignment)
            banana_figures= PlotsGenerator.getTrialSequencesArrayUsingClusters(banana, markers_and_colors, samples_with_clusters, alignment=alignment)
            lampa_figures= PlotsGenerator.getTrialSequencesArrayUsingClusters(lampa, markers_and_colors, samples_with_clusters, alignment=alignment)
            chitara_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(chitara, markers_and_colors, samples_with_clusters, alignment=alignment)
            masina_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(masina, markers_and_colors, samples_with_clusters, alignment=alignment)
            vaca_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(vaca, markers_and_colors, samples_with_clusters, alignment=alignment)
            furculita_figures= PlotsGenerator.getTrialSequencesArrayUsingClusters(furculita, markers_and_colors, samples_with_clusters, alignment=alignment)
            cerb_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(cerb, markers_and_colors, samples_with_clusters, alignment=alignment)
            pantaloni_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(pantaloni, markers_and_colors, samples_with_clusters, alignment=alignment)
            scaun_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(scaun, markers_and_colors, samples_with_clusters, alignment=alignment)
            peste_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(peste, markers_and_colors, samples_with_clusters, alignment=alignment)
            caine_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(caine, markers_and_colors, samples_with_clusters, alignment=alignment)
            sticla_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(sticla, markers_and_colors, samples_with_clusters, alignment=alignment)
            pistol_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(pistol, markers_and_colors, samples_with_clusters, alignment=alignment)
            bicicleta_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(bicicleta, markers_and_colors, samples_with_clusters,  alignment=alignment)
            cal_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(cal, markers_and_colors, samples_with_clusters, alignment=alignment)
            elefant_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(elicopter, markers_and_colors, samples_with_clusters, alignment=alignment)
            iepure_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(iepure, markers_and_colors, samples_with_clusters, alignment=alignment)
            pahar_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(pahar, markers_and_colors, samples_with_clusters, alignment=alignment)
            masa_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(masa, markers_and_colors, samples_with_clusters, alignment=alignment)
            umbrela_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(umbrela, markers_and_colors, samples_with_clusters, alignment=alignment)
            fluture_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(fluture, markers_and_colors, samples_with_clusters, alignment=alignment)
            girafa_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(girafa, markers_and_colors, samples_with_clusters, alignment=alignment)
            pian_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(pian, markers_and_colors, samples_with_clusters, alignment=alignment)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(poseta_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: poseta/geanta (de dama)\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_poseta.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(topor_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: topor/secure\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_topor.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(oala_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: oala/cratita\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_oala.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(elicopter_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: elicopter\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_elicopter.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(urs_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: urs (polar)\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_urs.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(palarie_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: palarie\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_palarie.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(foarfece_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: foarfece\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_foarfece.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(banana_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: banana\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_banana.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(lampa_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: lampa/veioza\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_lampa.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(chitara_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: chitara (electrica)\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_chitara.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(masina_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: masina\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_masina.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(vaca_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: vaca\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_vaca.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(furculita_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: furculita\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_furculita.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(cerb_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: cerb\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_cerb.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(pantaloni_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: pantaloni (scurti)\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_pantaloni.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(scaun_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: scaun\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_scaun.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(peste_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: peste\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_peste.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(caine_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: caine/catel\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_caine.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(sticla_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: sticla\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_sticla.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(pistol_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: pistol\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_pistol.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(bicicleta_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: bicicleta\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_bicicleta.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(cal_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: cal\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_cal.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(elefant_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: elefant\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_elefant.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(iepure_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: iepure\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_iepure.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(pahar_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: pahar/cupa\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_cupa.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(masa_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: masa\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_masa.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(umbrela_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: umbrela\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_umbrela.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(fluture_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: fluture\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_fluture.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(girafa_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: girafa\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_girafa.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(pian_figures, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: pian\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_pian.png", dpi=300)
        # plt.show()

        if method == Method.BMU:
            all_freq = [freq_poseta, freq_topor, freq_oala, freq_elicopter, freq_urs, freq_palarie, freq_foarfece,
                        freq_banana, freq_lampa, freq_chitara
                , freq_masina, freq_vaca, freq_furculita, freq_cerb, freq_pantaloni, freq_scaun, freq_peste, freq_caine,
                        freq_sticla
                , freq_pistol, freq_bicicleta, freq_cal, freq_elefant, freq_iepure, freq_pahar, freq_masa, freq_umbrela,
                        freq_fluture
                , freq_girafa, freq_pian]

            return all_freq


    @staticmethod
    def groupByVisibilityV2(all_trials_data, som, path, params, ssd=False, alignment=Alignment.LEFT, method=Method.BMU,
                          markers_and_colors=None, samples_with_clusters=None):
        v0_trials = all_trials_data[0:30]
        v1_trials = all_trials_data[30:60]
        v2_trials = all_trials_data[60:90]
        v3_trials = all_trials_data[90:120]
        v4_trials = all_trials_data[120:150]
        v5_trials = all_trials_data[150:180]
        v6_trials = all_trials_data[180:210]

        # sort after length
        v0_trials = PlotsGenerator.sortTrials(v0_trials)
        v1_trials = PlotsGenerator.sortTrials(v1_trials)
        v2_trials = PlotsGenerator.sortTrials(v2_trials)
        v3_trials = PlotsGenerator.sortTrials(v3_trials)
        v4_trials = PlotsGenerator.sortTrials(v4_trials)
        v5_trials = PlotsGenerator.sortTrials(v5_trials)
        v6_trials = PlotsGenerator.sortTrials(v6_trials)

        if method == Method.BMU:
            v0_figures, freq_v0 = PlotsGenerator.getTrialSequencesArrayUsingBMU(v0_trials, som,
                                                                                alignment=alignment)
            v1_figures, freq_v1 = PlotsGenerator.getTrialSequencesArrayUsingBMU(v1_trials, som,
                                                                                alignment=alignment)
            v2_figures, freq_v2 = PlotsGenerator.getTrialSequencesArrayUsingBMU(v2_trials, som,
                                                                                alignment=alignment)
            v3_figures, freq_v3 = PlotsGenerator.getTrialSequencesArrayUsingBMU(v3_trials, som,
                                                                                alignment=alignment)
            v4_figures, freq_v4 = PlotsGenerator.getTrialSequencesArrayUsingBMU(v4_trials, som,
                                                                                alignment=alignment)
            v5_figures, freq_v5 = PlotsGenerator.getTrialSequencesArrayUsingBMU(v5_trials, som,
                                                                                alignment=alignment)
            v6_figures, freq_v6 = PlotsGenerator.getTrialSequencesArrayUsingBMU(v6_trials, som,
                                                                                alignment=alignment)
        else:
            v0_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(v0_trials, markers_and_colors, samples_with_clusters, alignment=alignment)
            v1_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(v1_trials, markers_and_colors, samples_with_clusters, alignment=alignment)
            v2_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(v2_trials, markers_and_colors, samples_with_clusters,  alignment=alignment)
            v3_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(v3_trials, markers_and_colors, samples_with_clusters, alignment=alignment)
            v4_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(v4_trials, markers_and_colors, samples_with_clusters, alignment=alignment)
            v5_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(v5_trials, markers_and_colors, samples_with_clusters, alignment=alignment)
            v6_figures = PlotsGenerator.getTrialSequencesArrayUsingClusters(v6_trials, markers_and_colors, samples_with_clusters, alignment=alignment)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(v0_figures, n_rows=30, n_cols=1)
        plt.suptitle("Stimulus visibility: 0.00\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "v0.0.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(v1_figures, n_rows=30, n_cols=1)
        plt.suptitle("Stimulus visibility: 0.05\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "v0.05.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(v2_figures, n_rows=30, n_cols=1)
        plt.suptitle("Stimulus visibility: 0.1\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "v0.1.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(v3_figures, n_rows=30, n_cols=1)
        plt.suptitle("Stimulus visibility: 0.15\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "v0.15.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(v4_figures, n_rows=30, n_cols=1)
        plt.suptitle("Stimulus visibility: 0.2\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "v0.2.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(v5_figures, n_rows=30, n_cols=1)
        plt.suptitle("Stimulus visibility: 0.25\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "v0.25.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(v6_figures, n_rows=30, n_cols=1)
        plt.suptitle("Stimulus visibility: 0.3\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "v0.3.png", dpi=300)
        # plt.show()

        if method == Method.BMU:
            all_freq = [freq_v0, freq_v1, freq_v2, freq_v3, freq_v4, freq_v5, freq_v6]

            return all_freq

    # Helper methods for plots generation -----------------------------------------------------------------------------

    @staticmethod
    def getTrialSequencesArrayUsingBMU(all_trials_data, som, ssd=False, alignment=Alignment.LEFT):
        barcodes_array = []
        all_color_arrays = []
        max_length_color_array = 0
        color_frequencies_for_each_trial = []

        for cnt, trial in enumerate(all_trials_data):
            if cnt % 1000 == 0:
                print("Trial " + str(cnt))
            if ssd:
                colors_array, freq_matrix = Utils.get_colors_array(trial, som)
            else:
                colors_array, freq_matrix = Utils.get_colors_array(trial.trial_data, som)
            all_color_arrays.append(colors_array)
            color_frequencies_for_each_trial.append(freq_matrix)

            if len(colors_array) > max_length_color_array:
                max_length_color_array = len(colors_array)
        for cnt, colors_array in enumerate(all_color_arrays):
            if len(colors_array) != max_length_color_array:
                for i in range(0, max_length_color_array - len(colors_array)):
                    if alignment == Alignment.LEFT:
                        colors_array.append([1, 1, 1])
                    else:
                        colors_array.insert(0, [1, 1, 1])
            figure_data_tuple = PlotsGenerator.generateColorSequenceForTrialMatplotlib(colors_array)
            barcodes_array.append(figure_data_tuple)

        return barcodes_array, color_frequencies_for_each_trial

    @staticmethod
    def getTrialSequencesArrayUsingBMULeftAlignmentMINISOM(all_trials_data, som, ssd=False):
        barcodes_array = []
        all_color_arrays = []
        max_length_color_array = 0
        for cnt, trial in enumerate(all_trials_data):
            if cnt % 1000 == 0:
                print("Trial " + str(cnt))
            if ssd:
                colors_array = Utils.get_colors_arrayMINISOM(trial, som)
            else:
                colors_array = Utils.get_colors_arrayMINISOM(trial.trial_data, som)
            all_color_arrays.append(colors_array)
            if len(colors_array) > max_length_color_array:
                max_length_color_array = len(colors_array)
        for cnt, colors_array in enumerate(all_color_arrays):
            if len(colors_array) != max_length_color_array:
                for i in range(0, max_length_color_array - len(colors_array)):
                    colors_array.append([1, 1, 1])
            figure_data_tuple = PlotsGenerator.generateColorSequenceForTrialMatplotlib(colors_array)
            barcodes_array.append(figure_data_tuple)
        return barcodes_array

    @staticmethod
    def getTrialSequencesArrayUsingClusters(all_trials_data, markers_and_colors, samples_with_clusters,
                                            alignment=Alignment.LEFT):
        barcodes_array = []
        all_color_arrays = []
        max_length_color_array = 0
        index = 0
        for cnt, trial in enumerate(all_trials_data):
            print("Trial " + str(cnt))
            indexes_array = []
            for i, sample in enumerate(trial.trial_data):
                if i % 1000 == 0:
                    print("Sample progress " + str(i + index))
                indexes_array.append(i + index)
                # PlotsGenerator.validate(sample, samples_with_clusters[i+index][0])
                if i == len(trial.trial_data) - 1:
                    index = i + index + 1
            colors_array = Utils.get_colors_array_with_clusters(indexes_array, markers_and_colors,
                                                                samples_with_clusters)
            all_color_arrays.append(colors_array)
            if len(colors_array) > max_length_color_array:
                max_length_color_array = len(colors_array)
        for cnt, colors_array in enumerate(all_color_arrays):
            if len(colors_array) != max_length_color_array:
                for i in range(0, max_length_color_array - len(colors_array)):
                    if alignment == Alignment.LEFT:
                        colors_array.append([1, 1, 1])
                    else:
                        colors_array.insert(0, [1, 1, 1])
            figure_data_tuple = PlotsGenerator.generateColorSequenceForTrialMatplotlib(colors_array)
            barcodes_array.append(figure_data_tuple)
        return barcodes_array

    @staticmethod
    def generateColorSequenceForTrialMatplotlib(colors, width=400, height=100):
        color_indices = np.arange(len(colors))
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow([color_indices], cmap=cmap, aspect="auto", extent=(0, width, 0, height))
        ax.set_xticks([])
        ax.set_yticks([])
        # print('Barcode generated')
        plt.close()
        return fig, ax, color_indices

    @staticmethod
    def generateGridWithColorSequences(figure_data_array, n_rows=2, n_cols=1, width=400, height=100):
        figure_index = 0
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
        for i in range(n_rows * n_cols):
            # print('Barcode grid figure index ', figure_index)
            row_idx = i // n_cols
            col_idx = i % n_cols
            _, ax, color_indices = figure_data_array[figure_index]
            figure_index += 1
            axs[row_idx, col_idx].axis('off')
            axs[row_idx, col_idx].imshow(color_indices.reshape(1, -1), cmap=ax.images[0].cmap, aspect='auto')
        plt.subplots_adjust(hspace=0, wspace=0)
        return fig, axs

    # Scatter plot -------------------------------------------------------------------------------------------

    @staticmethod
    def generateScatterPlotForDistanceMapPlotly(size, distance_map):
        colors = []
        X_coord = []
        Y_coord = []
        Z_coord = []

        for i in range(0, size):
            for j in range(0, size):
                for k in range(0, size):
                    X_coord.append(i)
                    Y_coord.append(j)
                    Z_coord.append(k)
                    colors.append(distance_map[i][j][k])

        trace = go.Scatter3d(
            x=X_coord,
            y=Y_coord,
            z=Z_coord,
            mode='markers',
            marker=dict(
                size=10,
                color=colors,
                opacity=0.8,
                symbol='circle',
                colorscale='Viridis',
                showscale=True
            ),
            text=colors
        )

        layout = go.Layout(
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
            )
        )

        # Create a figure with the trace and layout, and show the plot
        fig1 = go.Figure(data=[trace], layout=layout)
        fig1.update_layout(
            title='Distance map'
        )
        fig1.show()

    @staticmethod
    def generateScatterPlotForClustersPlotly(som, processed_data):
        threshold = som.find_threshold(processed_data)
        print('Max dist ', threshold)
        no_clusters, bmu_array, samples_with_clusters_array = som.find_clusters_with_min_dist(
            processed_data,
            0.3, threshold)
        print('No clusters ', no_clusters)
        samples_with_symbols_array = Utils.assign_symbols(samples_with_clusters_array)
        print(samples_with_symbols_array)

        markers_and_colors = Utils.assign_markers_and_colors(no_clusters)
        print(no_clusters)
        print("-------------------------------------")

        BMU_X = []
        BMU_Y = []
        BMU_Z = []
        M = []
        C = []
        BMU = []

        for cnt, xx in enumerate(processed_data):
            w = som.find_BMU(xx)
            # print('W ', w)
            cluster = 0
            for bmu in bmu_array:
                if w == bmu[0]:
                    cluster = bmu[1]
                    break
            marker = '_'
            color = 'y'
            for x in markers_and_colors:
                if x[0] == cluster:
                    marker = x[1]
                    color = x[2]
            notInBMU = True
            for x in BMU:
                if w == x:
                    notInBMU = False
            if notInBMU:
                BMU.append(w)
                BMU_X.append(w[0])
                BMU_Y.append(w[1])
                BMU_Z.append(w[2])
                M.append(marker)
                C.append(color)

        print(M)

        trace = go.Scatter3d(
            x=BMU_X,
            y=BMU_Y,
            z=BMU_Z,
            mode='markers',
            marker=dict(
                size=5,
                color=C,
                opacity=0.8,
                symbol=M
            )
        )

        layout = go.Layout(
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
            )
        )

        # Create a figure with the trace and layout, and show the plot
        fig = go.Figure(data=[trace], layout=layout)
        fig.update_layout(
            title='Clusters given by som'
        )
        fig.show()
        return samples_with_clusters_array, markers_and_colors

    @staticmethod
    def generateScatterPlotForDistanceMapMatplotlib(size, distance_map):
        Nx, Ny, Nz = size, size, size
        X, Y, Z = np.arange(Nx), np.arange(Ny), -np.arange(Nz)

        fig = plt.figure(figsize=(size, size))
        ax = Axes3D(fig)

        # Add x, y gridlines
        ax.grid(b=True, color='red',
                linestyle='-.', linewidth=0.3,
                alpha=0.2)
        kw = {
            'cmap': 'Blues'
        }
        for i in range(0, size):
            for j in range(0, size):
                for k in range(0, size):
                    ax.scatter3D(X[i], Y[j], Z[k], c=str(distance_map[i][j][k]), s=100, cmap=plt.get_cmap('jet'))

        # Show Figure
        plt.show()

    @staticmethod
    def generateScatterPlotForClustersMatplotlib(som, eegDataProcessor, size):
        threshold = som.find_threshold(eegDataProcessor.processed_data)
        print('Max dist ', threshold)
        no_clusters, bmu_array, samples_with_clusters_array = som.find_clusters_with_min_dist(
            eegDataProcessor.processed_data,
            0.3, threshold)
        print('No clusters ', no_clusters)

        fig = plt.figure(figsize=(size, size))
        ax = Axes3D(fig)
        ax.grid(b=True, color='red',
                linestyle='-.', linewidth=0.3,
                alpha=0.2)
        markers_and_colors = Utils.assign_markers_and_colors(no_clusters)
        for cnt, xx in enumerate(eegDataProcessor.processed_data):
            w = som.find_BMU(xx)
            cluster = 0
            for bmu in bmu_array:
                if w == bmu[0]:
                    cluster = bmu[1]
                    break
            marker = '_'
            color = 'y'
            for x in markers_and_colors:
                if x[0] == cluster:
                    marker = x[1]
                    color = x[2]
            ax.scatter3D(w[0], w[1], w[2], color=color, marker=marker)
        # Show Figure
        plt.show()

    # Volume slice plot Mayavi ----------------------------------------------------------------------------------

    @staticmethod
    def generateSlicerPlotMayavi(distance_map):
        volume_slice_x = mlab.volume_slice(distance_map, plane_orientation='x_axes')
        volume_slice_y = mlab.volume_slice(distance_map, plane_orientation='y_axes')
        volume_slice_z = mlab.volume_slice(distance_map, plane_orientation='z_axes')
        outline = mlab.outline(volume_slice_x)
        colorbar = mlab.colorbar(object=volume_slice_x, title='Data values')
        mlab.show()

    # NOT USED ---------------------------------------------------------------------------------------------------

    @staticmethod
    def groupByResponseV1(figure_array, path, params):
        nothing_figures = PlotsGenerator.getNothingData(figure_array)
        something_figures = PlotsGenerator.getSomethingData(figure_array)
        identified_figures = PlotsGenerator.getIdentifiedData(figure_array)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(nothing_figures, n_rows=len(nothing_figures),
                                                                n_cols=1)
        plt.suptitle("Response: nothing\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "response_nothing.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(something_figures, n_rows=len(something_figures),
                                                                n_cols=1)
        plt.suptitle("Response: something\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "response_something.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(identified_figures, n_rows=len(identified_figures),
                                                                n_cols=1)
        plt.suptitle("Response: what the subject sees (correct + 1 incorrect)\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "response_identified.png", dpi=300)
        plt.show()

    # visibilities: 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30
    @staticmethod
    def groupByVisibilityV1(figure_array, path, params):
        fig, ax = PlotsGenerator.generateGridWithColorSequences(figure_array[0:30], n_rows=30, n_cols=1)
        plt.suptitle("Stimulus visibility: 0.00\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "v0.0.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(figure_array[30:60], n_rows=30, n_cols=1)
        plt.suptitle("Stimulus visibility: 0.05\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "v0.05.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(figure_array[60:90], n_rows=30, n_cols=1)
        plt.suptitle("Stimulus visibility: 0.1\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "v0.1.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(figure_array[90:120], n_rows=30, n_cols=1)
        plt.suptitle("Stimulus visibility: 0.15\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "v0.15.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(figure_array[120:150], n_rows=30, n_cols=1)
        plt.suptitle("Stimulus visibility: 0.2\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "v0.2.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(figure_array[150:180], n_rows=30, n_cols=1)
        plt.suptitle("Stimulus visibility: 0.25\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "v0.25.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(figure_array[180:210], n_rows=30, n_cols=1)
        plt.suptitle("Stimulus visibility: 0.3\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "v0.3.png", dpi=300)
        plt.show()

    @staticmethod
    def groupByStimulusV1(figure_array, path, params):
        poseta = PlotsGenerator.getStimulusPosetaData(figure_array)
        topor = PlotsGenerator.getStimulusToporData(figure_array)
        oala = PlotsGenerator.getStimulusOalaData(figure_array)
        elicopter = PlotsGenerator.getStimulusElicopterData(figure_array)
        urs = PlotsGenerator.getStimulusUrsData(figure_array)
        palarie = PlotsGenerator.getStimulusPalarieData(figure_array)
        foarfece = PlotsGenerator.getStimulusFoarfeceData(figure_array)
        banana = PlotsGenerator.getStimulusBananaData(figure_array)
        lampa = PlotsGenerator.getStimulusLampaData(figure_array)
        chitara = PlotsGenerator.getStimulusChitaraData(figure_array)
        masina = PlotsGenerator.getStimulusMasinaData(figure_array)
        vaca = PlotsGenerator.getStimulusVacaData(figure_array)
        furculita = PlotsGenerator.getStimulusFurculitaData(figure_array)
        cerb = PlotsGenerator.getStimulusCerbData(figure_array)
        pantaloni = PlotsGenerator.getStimulusPantaloniData(figure_array)
        scaun = PlotsGenerator.getStimulusScaunData(figure_array)
        peste = PlotsGenerator.getStimulusPesteData(figure_array)
        caine = PlotsGenerator.getStimulusCaineData(figure_array)
        sticla = PlotsGenerator.getStimulusSticlaData(figure_array)
        pistol = PlotsGenerator.getStimulusPistolData(figure_array)
        bicicleta = PlotsGenerator.getStimulusBicicletaData(figure_array)
        cal = PlotsGenerator.getStimulusCalData(figure_array)
        elefant = PlotsGenerator.getStimulusElefantData(figure_array)
        iepure = PlotsGenerator.getStimulusIepureData(figure_array)
        pahar = PlotsGenerator.getStimulusPaharData(figure_array)
        masa = PlotsGenerator.getStimulusMasaData(figure_array)
        umbrela = PlotsGenerator.getStimulusUmbrelaData(figure_array)
        fluture = PlotsGenerator.getStimulusFlutureData(figure_array)
        girafa = PlotsGenerator.getStimulusGirafaData(figure_array)
        pian = PlotsGenerator.getStimulusPianData(figure_array)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(poseta, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: poseta/geanta (de dama)\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_poseta.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(topor, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: topor/secure\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_topor.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(oala, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: oala/cratita\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_oala.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(elicopter, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: elicopter\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_elicopter.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(urs, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: urs (polar)\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_urs.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(palarie, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: palarie\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_palarie.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(foarfece, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: foarfece\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_foarfece.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(banana, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: banana\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_banana.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(lampa, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: lampa/veioza\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_lampa.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(chitara, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: chitara (electrica)\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_chitara.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(masina, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: masina\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_masina.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(vaca, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: vaca\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_vaca.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(furculita, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: furculita\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_furculita.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(cerb, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: cerb\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_cerb.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(pantaloni, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: pantaloni (scurti)\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_pantaloni.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(scaun, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: scaun\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_scaun.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(peste, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: peste\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_peste.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(caine, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: caine/catel\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_caine.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(sticla, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: sticla\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_sticla.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(pistol, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: pistol\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_pistol.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(bicicleta, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: bicicleta\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_bicicleta.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(cal, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: cal\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_cal.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(elefant, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: elefant\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_elefant.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(iepure, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: iepure\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_iepure.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(pahar, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: pahar/cupa\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_cupa.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(masa, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: masa\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_masa.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(umbrela, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: umbrela\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_umbrela.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(fluture, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: fluture\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_fluture.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(girafa, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: girafa\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_girafa.png", dpi=300)

        fig, ax = PlotsGenerator.generateGridWithColorSequences(pian, n_rows=7, n_cols=1)
        plt.suptitle("Stimulus: pian\n" + params)
        fig.set_size_inches(6, 4)
        plt.savefig(path + "stimulus_pian.png", dpi=300)
        plt.show()

    @staticmethod
    def generateColorSeguenceForAllTrialsWithImages(no_trials, all_trials_data, som):
        fig = plt.figure(figsize=(10, 10))
        rows = no_trials
        columns = 1
        for cnt, trial in enumerate(all_trials_data):
            if cnt % 1000 == 0:
                print("Trial " + str(cnt))
            color_seq_fig, color_sequence = PlotsGenerator.generateColorSequenceForTrial(cnt, trial.trial_data, som)
            image_bytes = pio.to_image(color_seq_fig, format='png')
            image = Image.open(io.BytesIO(image_bytes))
            fig.add_subplot(rows, columns, cnt)
            plt.imshow(image)
            plt.axis('off')
            plt.title("Trial " + str(cnt + 1))
        fig.show()

    @staticmethod
    def generateColorSeguenceForAllTrialsInPDF(no_trials, all_trials_data, som):
        PAGE_WIDTH, PAGE_HEIGHT = A4
        c = canvas.Canvas("trials.pdf", pagesize=A4)
        TOP_MARGIN = 0
        BOTTOM_MARGIN = 0
        img_height = 280
        y = PAGE_HEIGHT - TOP_MARGIN - img_height
        for cnt, trial in enumerate(all_trials_data):
            if cnt % 1000 == 0:
                print("Trial " + str(cnt))
            color_seq_fig, color_sequence = PlotsGenerator.generateColorSequenceForTrial(cnt, trial.trial_data, som)
            image_bytes = pio.to_image(color_seq_fig, format='png')
            image = Image.open(io.BytesIO(image_bytes))
            if y >= BOTTOM_MARGIN:
                print("same")
                c.drawImage(ImageReader(image), 10, y, width=600, height=280, preserveAspectRatio=True)
                y -= img_height
            else:
                print("other")
                c.showPage()
                y = PAGE_HEIGHT - TOP_MARGIN - img_height
                c.drawImage(ImageReader(image), 10, y, width=600, height=280, preserveAspectRatio=True)
                y -= img_height
        c.save()

    @staticmethod
    def validate(a1, a2):
        for cnt, el1 in enumerate(a1):
            if el1 != a2[cnt]:
                print("False")
                break
        print("True")

    @staticmethod
    def generateColorSeguenceForAllTrials(no_trials, all_trials_data, som):

        fig = make_subplots(rows=no_trials, cols=1)

        color_seg_lengths = []

        for cnt, trial in enumerate(all_trials_data):
            color_seq_fig, color_sequence = PlotsGenerator.generateColorSequenceForTrial(cnt, trial, som)
            color_seg_lengths.append(len(color_sequence))
            for shape in color_seq_fig.layout.shapes:
                fig.add_shape(shape, row=cnt + 1, col=1)

        for i in range(no_trials):
            fig.update_xaxes(title_text="Time", range=[0, color_seg_lengths[i]], row=i + 1, col=1)
            fig.update_yaxes(title_text="Trial " + str(i + 1), range=[0, 3], row=i + 1, col=1)

        fig.update_layout(title='Color sequences for each trial')

        fig.show()

    @staticmethod
    def generateColorSequenceForTrial(no_trial, trial_data, som):
        colors_sequence = Utils.get_rgb_colors_array(trial_data, som)
        fig = go.Figure()
        # Add a rectangle trace for each segment of the barcode
        for i in range(len(colors_sequence)):
            print('Cs ', i)
            fig.add_shape(
                type='rect',
                x0=i,
                y0=0,
                x1=i + 1,
                y1=3,
                fillcolor=colors_sequence[i],
                line=dict(width=0),
            )

        # Update the layout
        fig.update_layout(
            xaxis=dict(range=[0, len(colors_sequence)], title='time', showticklabels=False),
            yaxis=dict(title='Trial ' + no_trial, range=[0, 3]),
            width=1200,
        )

        return fig, colors_sequence

    """
    @staticmethod
    def computePSIByResponse(color_freq_for_each_trial):
        nothing_color_freq = PlotsGenerator.getNothingData(color_freq_for_each_trial)

        something_color_freq = PlotsGenerator.getSomethingData(color_freq_for_each_trial)

        identified_color_freq = PlotsGenerator.getIdentifiedData(color_freq_for_each_trial)

        nothing_total_freq_matrix = PlotsGenerator.computeTotalFrequenciesByGroup(nothing_color_freq)
        something_total_freq_matrix = PlotsGenerator.computeTotalFrequenciesByGroup(something_color_freq)
        identified_total_freq_matrix = PlotsGenerator.computeTotalFrequenciesByGroup(identified_color_freq)

        total_freq_matrix = nothing_total_freq_matrix + something_total_freq_matrix + identified_total_freq_matrix

        nothing_PSIs_for_all_colors_matrix = PlotsGenerator.computePSI(nothing_total_freq_matrix, total_freq_matrix)
        something_PSIs_for_all_colors_matrix = PlotsGenerator.computePSI(something_total_freq_matrix, total_freq_matrix)
        identified_PSIs_for_all_colors_matrix = PlotsGenerator.computePSI(identified_total_freq_matrix,
                                                                          total_freq_matrix)
        return nothing_PSIs_for_all_colors_matrix, something_PSIs_for_all_colors_matrix, identified_PSIs_for_all_colors_matrix

    @staticmethod
    def computePSIByVisibility(color_freq_for_each_trial):
        v0_color_freq = color_freq_for_each_trial[0:30]
        v1_color_freq = color_freq_for_each_trial[30:60]
        v2_color_freq = color_freq_for_each_trial[60:90]
        v3_color_freq = color_freq_for_each_trial[90:120]
        v4_color_freq = color_freq_for_each_trial[120:150]
        v5_color_freq = color_freq_for_each_trial[150:180]
        v6_color_freq = color_freq_for_each_trial[180:210]

        v0_total_freq_matrix = PlotsGenerator.computeTotalFrequenciesByGroup(v0_color_freq)
        v1_total_freq_matrix = PlotsGenerator.computeTotalFrequenciesByGroup(v1_color_freq)
        v2_total_freq_matrix = PlotsGenerator.computeTotalFrequenciesByGroup(v2_color_freq)
        v3_total_freq_matrix = PlotsGenerator.computeTotalFrequenciesByGroup(v3_color_freq)
        v4_total_freq_matrix = PlotsGenerator.computeTotalFrequenciesByGroup(v4_color_freq)
        v5_total_freq_matrix = PlotsGenerator.computeTotalFrequenciesByGroup(v5_color_freq)
        v6_total_freq_matrix = PlotsGenerator.computeTotalFrequenciesByGroup(v6_color_freq)


        total_freq_matrix = v0_total_freq_matrix + v1_total_freq_matrix + v2_total_freq_matrix + v3_total_freq_matrix + v4_total_freq_matrix + v5_total_freq_matrix + v6_total_freq_matrix
        v0_PSIs_for_all_colors_matrix = PlotsGenerator.computePSI(v0_total_freq_matrix, total_freq_matrix)
        v1_PSIs_for_all_colors_matrix = PlotsGenerator.computePSI(v1_total_freq_matrix, total_freq_matrix)
        v2_PSIs_for_all_colors_matrix = PlotsGenerator.computePSI(v2_total_freq_matrix, total_freq_matrix)
        v3_PSIs_for_all_colors_matrix = PlotsGenerator.computePSI(v3_total_freq_matrix, total_freq_matrix)
        v4_PSIs_for_all_colors_matrix = PlotsGenerator.computePSI(v4_total_freq_matrix, total_freq_matrix)
        v5_PSIs_for_all_colors_matrix = PlotsGenerator.computePSI(v5_total_freq_matrix, total_freq_matrix)
        v6_PSIs_for_all_colors_matrix = PlotsGenerator.computePSI(v6_total_freq_matrix, total_freq_matrix)

        return v0_PSIs_for_all_colors_matrix, v1_PSIs_for_all_colors_matrix, v2_PSIs_for_all_colors_matrix, v3_PSIs_for_all_colors_matrix, v4_PSIs_for_all_colors_matrix, v5_PSIs_for_all_colors_matrix, v6_PSIs_for_all_colors_matrix


    @staticmethod
    def generateGridWithColorSequencesWithPSI(figure_data_array, PSI_matrix_for_all_colors, threshold, n_rows=2,
                                              n_cols=1, width=400, height=100):
        figure_index = 0
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
        for i in range(n_rows * n_cols):
            print('Barcode grid figure index ', figure_index)
            row_idx = i // n_cols
            col_idx = i % n_cols
            _, ax, color_indices = figure_data_array[figure_index]
            figure_index += 1
            axs[row_idx, col_idx].axis('off')
            axs[row_idx, col_idx].imshow(color_indices.reshape(1, -1), cmap=ax.images[0].cmap, aspect='auto')
        plt.subplots_adjust(hspace=0, wspace=0)
        return fig, axs

    """
