import click
import os
from pathlib import Path


def arg_parse(ctx: 'click.Context') -> dict:
    # 解析额外参数
    extra_kwargs = {}
    i = 0
    while i < len(ctx.args):
        arg = ctx.args[i]
        if arg.startswith('--'):
            param_name = arg[2:].replace('-', '_')  # 转换参数名格式
            i += 1
            if i < len(ctx.args) and not ctx.args[i].startswith('--'):
                # 参数有值
                try:
                    # 尝试转换为适当的类型
                    if ctx.args[i].lower() == 'true':
                        extra_kwargs[param_name] = True
                    elif ctx.args[i].lower() == 'false':
                        extra_kwargs[param_name] = False
                    elif '.' in ctx.args[i]:
                        try:
                            extra_kwargs[param_name] = float(ctx.args[i])
                        except ValueError:
                            extra_kwargs[param_name] = ctx.args[i]
                    else:
                        try:
                            extra_kwargs[param_name] = int(ctx.args[i])
                        except ValueError:
                            extra_kwargs[param_name] = ctx.args[i]
                except:
                    extra_kwargs[param_name] = ctx.args[i]

                if param_name in {'config_path', 'config'} and extra_kwargs.get(param_name):
                    os.environ['MINERU_TOOLS_CONFIG_JSON'] = str(
                        Path(str(extra_kwargs[param_name])).expanduser().resolve()
                    )
                    extra_kwargs.pop(param_name, None)
            else:
                # 布尔型标志参数
                extra_kwargs[param_name] = True
                i -= 1
        i += 1
    return extra_kwargs