class Utils:
    all_markers = ['circle', 'square', 'cross', 'diamond',
                   'diamond-open', 'circle-open', 'square-open', 'x']
    all_colors = ['blue', 'fuchsia', 'gold', 'green', 'violet', 'purple', 'red', 'royalblue', 'navy',
                  'mediumspringgreen']

    all_symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']

    @staticmethod
    def assign_markers_and_colors(no_clusters):  # for scatter plot
        markers_and_colors = []
        for i in range(no_clusters):
            markers_and_colors.append(
                (i, Utils.all_markers[i % len(Utils.all_markers)], Utils.all_colors[i % len(Utils.all_colors)]))
        return markers_and_colors

    @staticmethod
    def assign_symbols(samples_with_clusters):  # for scatter plot
        samples_with_symbols = []
        for (sample, cluster) in samples_with_clusters:
            samples_with_symbols.append(
                (sample, Utils.all_symbols[cluster]))
        return samples_with_symbols

    @staticmethod
    def get_colors_array(samples, som):
        colors_array = []
        for s in samples:
            bmu = som.find_BMU(s)
            colors_array.append([float(bmu[0]) / float(som.getX()), float(bmu[1]) / float(som.getY()),
                                 float(bmu[2]) / float(som.getZ())])
        return colors_array

    @staticmethod
    def get_rgb_colors_array(samples, som):
        colors_array = []
        for s in samples:
            bmu = som.find_BMU(s)
            color_r = (float(bmu[0]) / float(som.getX())) * 255
            color_g = (float(bmu[1]) / float(som.getY())) * 255
            color_b = (float(bmu[2]) / float(som.getZ())) * 255
            rgb_string = 'rgb(' + str(color_r)+','+str(color_g) + ',' + str(color_b) + ')'
            colors_array.append(rgb_string)
        return colors_array
