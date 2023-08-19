# Panel 库示例
# pip install panel
import panel as pn
from QA import  QA
import time
pn.extension()

panels = [pn.Row('Assistant:', pn.pane.Markdown("Hello! I am AI assistant for arVix.", width=600, styles={'background-color': '#F6F6F6'}))] # 定义预置欢迎语句

inp = pn.widgets.TextInput(value="", placeholder='You can ask me anything you want') # 输入框
button_conversation = pn.widgets.Button(name="Ask arVix") # 提问按钮
button_clear_history = pn.widgets.Button(name="Clear history") # 清除历史按钮
qa = QA()

async def collect_messages(event1,event2): # 并行推断回调
    global panels, inp, interactive_conversation,qa # 调用全局变量
    if inp.value_input == "": # 如果输入为空
        yield pn.Column(*panels)
    if event1: # button_conversation 事件
        prompt = inp.value_input # 获取用户输入
        inp.value_input = '' # 清空输入框
        panels.append(pn.Row('User:', pn.pane.Markdown(prompt, width=600)))
        yield pn.Column(*panels) # 并行更新用户输入
        # 过程提示输出示例
        for i in range(5):
            time.sleep(1)
            panels.append(pn.Row('System:', pn.pane.Markdown(f"wait {i+1}/5", width=600, styles={'background-color': '#F6F6F6'})))
            yield pn.Column(*panels) # 返回过程提示
        for i in range(5):
            panels.pop(-1) # 清空过程提示
        # 过程提示输出示例结束
        response = qa.get_response(prompt) # 获取AI输出
        panels.append(pn.Row('Assistant:', pn.pane.Markdown(response, width=600, styles={'background-color': '#F6F6F6'})))
    elif event2: # button_clear_history 事件
        panels = [pn.Row('Assistant:', pn.pane.Markdown("Hello! I am AI assistant for arVix.", width=600, styles={'background-color': '#F6F6F6'}))] # 重置
    yield pn.Column(*panels)

interactive_conversation = pn.bind(collect_messages, button_conversation, button_clear_history) #链接函数与按钮组合

dashboard = pn.Column( #定义显示结构
    inp,
    pn.Row(button_conversation,button_clear_history),
    pn.panel(interactive_conversation, loading_indicator=True, height=300),
)

dashboard.servable()


# import panel as pn
# import hvplot.pandas
# from bokeh.sampledata.autompg import autompg
#
# columns = list(autompg.columns[:-2])
#
# x = pn.widgets.Select(value='mpg', options=columns, name='x')
# y = pn.widgets.Select(value='hp', options=columns, name='y')
# color = pn.widgets.ColorPicker(name='Color', value='#AA0505')
#
# pn.Row(
#     pn.Column('## MPG Explorer', x, y, color),
#     pn.bind(autompg.hvplot.scatter, x, y, c=color)
# ).servable()