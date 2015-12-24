from bokeh.io import vplot
from bokeh.models import ColumnDataSource, DataRange1d, Plot, LinearAxis, Grid, Circle, HoverTool, BoxSelectTool
from bokeh.models.widgets import DataTable, TableColumn, StringFormatter, NumberFormatter, StringEditor, IntEditor, NumberEditor, SelectEditor
from bokeh.embed import file_html
from bokeh.resources import INLINE
from bokeh.browserlib import view
from bokeh.sampledata.autompg2 import autompg2 as mpg
from bokeh.plotting import figure, show, output_file
from bokeh.io import output_file, show, vplot

from collections import defaultdict


def save_reports(reports):
    print(len(reports))
    output_file("out.html")
    cols = ['model', 'dataset', 'nb_examples_train',
            'nb_examples_test', 'seed', u'duration',
            'accuracy_train', 'accuracy_test']
    cols_hp = reports[0]["hp"].keys()
    data = defaultdict(list)
    for i, r in enumerate(reports):
        for c in cols:
            if c in r:
                if c == "start" or c == "end":
                    o = str(r[c])
                elif type(r[c]) == list:
                    o = r[c][-1]
                else:
                    o = r[c]
                data[c].append(o)
            else:
                data[c].append(None)
        for c in cols_hp:
            data[c].append(r["hp"].get(c))
        data["id"].append(i)

    source = ColumnDataSource(data)
    columns = [TableColumn(field="id", title="id")] + [
             TableColumn(field=c, title=c)
             for c in cols + cols_hp
    ]
    data_table = DataTable(source=source, columns=columns, editable=False)
    P = []

    for i, r in enumerate(reports):
        if "accuracy_train" not in r or "accuracy_test" not in r:
            continue
        epochs = range(1, len(r["accuracy_train"]) + 1)
        p = figure(title="accuracy exp {0}".format(i))
        p.line(epochs, r["accuracy_train"], line_width=2, color="blue", legend="train")
        p.line(epochs, r["accuracy_test"], line_width=2, color="green", legend="test")
        P.append(p)

    v = vplot(data_table, *P)
    show(v)

if __name__ == "__main__":

    from lightexperiments.light import Light
    light = Light()
    light.launch()
    reports = light.db.find({"tags": "deepconvnets"})
    save_reports(list(reports))
    light.close()
