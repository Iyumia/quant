import pyecharts.options as opts
from abc import abstractmethod, ABCMeta
from pyecharts.components import Table
from pyecharts.charts import Bar, Line, Kline, Page
from pyecharts.commons.utils import JsCode
from pyecharts.globals import ThemeType, CurrentConfig, NotebookType  
from pandas import DataFrame

# global setting
CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_LAB

class eChart(object):
    """ abstractmethod of pyecharts object """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, data:DataFrame, title:str, subtitle:str) -> None:
        pass

    def init(self):
        return self.chart.load_javascript()

    def show(self):
        return self.chart.render_notebook()

    def save(self, html):
        self.chart.load_javascript()
        return self.chart.render(html)

class eTable(eChart):
    """ pyecharts Table object """
    def __init__(self, data, title, subtitle) -> None:
        self.title = title
        self.subtitle = subtitle
        self.chart = Table()
        self.headers = data.columns.to_list()
        self.rows = list(data.values)
        self.chart.add(self.headers, self.rows)
        self.chart.set_global_opts({"title":self.title, "subtitle":self.subtitle,"title_style":"style='color:red'","subtitle_style":"style='color:green'"})

class eBar(eChart):
    """pyecharts Bar object"""
    def __init__(self, data, title, subtitle) -> None:
        self.chart = Bar({"theme": ThemeType.MACARONS})
        self.axis = [(label, series.to_list()) for (label, series) in data.items()]

        for i in range(len(self.axis)):
            if i == 0:
                self.chart.add_xaxis(self.axis[i][1])
            else:
                self.chart.add_yaxis(*self.axis[i])

        self.chart.set_global_opts(
            title_opts=opts.TitleOpts(title=title, subtitle=subtitle),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=True),
            )

class eLine(eChart):
    """ pyehcarts Line object"""
    def __init__(self, data, title, subtitle) -> None:
        self.chart = Line({"theme": ThemeType.MACARONS})
        self.axis = [(label, series.to_list()) for (label, series) in data.items()]

        for i in range(len(self.axis)):
            if i == 0:
                self.chart.add_xaxis(self.axis[i][1])
            else:
                self.chart.add_yaxis(
                    series_name = self.axis[i][0],
                    is_smooth = True,
                    y_axis = self.axis[i][1],
                    label_opts=opts.LabelOpts(is_show=False),
                    linestyle_opts=opts.LineStyleOpts(width=1),
                    )

        self.chart.set_global_opts(
            title_opts=opts.TitleOpts(title=title, subtitle=subtitle),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
            )

class eKline(eChart):
    def __init__(self, data, title, subtitle='Kline'):
        self.chart = Kline()
        self.xaxis = data.date
        self.yaxis = data.get(['open','high','low','close']).values
        self.chart.add_xaxis(self.xaxis)
        self.chart.add_yaxis(subtitle, self.yaxis)

        self.chart.set_global_opts(
        xaxis_opts=opts.AxisOpts(is_scale=True),
        yaxis_opts=opts.AxisOpts(
            is_scale=True,
            splitarea_opts=opts.SplitAreaOpts(
                is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
            ),
        ),
        datazoom_opts=[opts.DataZoomOpts(type_="inside")],
        title_opts=opts.TitleOpts(title=title),
        )

class ePage(eChart):
    """ pyecharts Page object """
    def __init__(self, dynamic=False) -> None:
        if dynamic:
            self.chart = Page(layout=Page.DraggablePageLayout)
        else:
            self.chart = Page()

    def add(self, *echarts):
        for c in echarts:
            self.chart.add(c.chart)


