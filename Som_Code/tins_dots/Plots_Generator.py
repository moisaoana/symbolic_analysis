import plotly.graph_objs as go
from plotly.subplots import make_subplots

from Som_Code.utils import Utils


class PlotsGenerator:

    @staticmethod
    def generateColorSequenceForTrial(no_trial, trial_data, som):
        colors_sequence = Utils.get_rgb_colors_array(trial_data, som)

        fig = go.Figure()

        # Add a rectangle trace for each segment of the barcode
        for i in range(len(colors_sequence)):
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
            fig.update_xaxes(title_text="Time", range=[0, color_seg_lengths[i]], row=i+1, col=1)
            fig.update_yaxes(title_text="Trial " + str(i+1), range=[0, 3], row=i+1, col=1)

        fig.update_layout(title='Color sequences for each trial')

        fig.show()
