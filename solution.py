import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

from openai import AsyncOpenAI


def read_task(path: Path) -> str:
    return path.read_text(encoding="utf-8")

client = OpenAI()

tools = [
] #пока оставляю пустым не разобрался тута из за адунов на оргах


#художественный фильм с их baseline.py
async def run_agent(task_path: Path):
    task_path = task_path.resolve()
    task = read_task(task_path)
    model = os.environ["OPENAI_MODEL"]

    #наш системный промпт
    systemPrompt = ("""
        Ты - профессиональный автономный Data Science агент. У тебя нет доступа к интернету.
        Ты имеешь доступ исключительно к каталогу в котором находится файл solution.py.
        Все зависимости, которые могли быть использованы описаны в файле requirements.txt.
        Твоя задача - на основе задачи, которая будет подана пользователем построить ML-модель.
        При этом тебе нужно самостоятельно по данным, которые будут предоставлены в .csv файлах определить тип задачи.
        Файл с предсказаниями submission.csv обязательно сохрани в текущий каталог.
        Строгие правила:
        1)Изучи структуру входных данных, определи тип ML-задачи и требования к результату.
        2)Оценивай качество модели с помощью собственной валидации для формирования итогового предсказания.
        3)Используй цикл AutoResearch для анализа эксперементов, формирования гипотез, проверки их и выбора наилучшего варианта.
        4)Не спеши с конечным результатом. Твоё время работы составляет 10800 секунд с момента старта файла solution.py. Используй время по максимуму для достижения наивысшей метрики.
        5)Но твоя цель это создание и с каждой итерации улучшение ML-модели. Не преследуй цель улучшения метрики любым другим путём кроме повышения точности ML-моделию.
        6)Учти что после получения файла с заданием и запуска solution.py соотвественно ты больше получишь указаний по действиям.
        7)Ты будешь действовать полностью автономно и не будут дополнительных подсказок по созданию по ходу твоей работы, поэтому анализируй тщательно.
    """)

    messages = [
        {"role": "system", "content": systemPrompt},
        {"role": "user", "content": task},
    ]
    server = StdioServerParameters( #сюда не смотри я сам хз на что тут заменить
        command=sys.executable,
        args=[str(Path(__file__).resolve()), "--mcp-server"],
        cwd=str(task_path.parent),
    )
    async with AsyncOpenAI() as client, stdio_client(server) as (read, write): #и сюда тоже
        async with ClientSession(read, write) as session:
            await session.initialize()
            available = await session.list_tools()
            tools = [{"type": "function", "function": {
                "name": tool.name, "description": tool.description,
                "parameters": tool.inputSchema,
            }} for tool in available.tools]
            for _ in range(100):
                response = await client.chat.completions.create(
                    model=model, messages=messages, tools=tools, parallel_tool_calls=False,
                )
                message = response.choices[0].message
                messages.append(message.model_dump(exclude_none=True))
                if not message.tool_calls:
                    print(message.content or "Агент завершил работу.")
                    return
                for call in message.tool_calls:
                    try:
                        result = await session.call_tool(call.function.name, json.loads(call.function.arguments))
                        output = "\n".join(block.text for block in result.content if block.type == "text")
                    except ValueError as error:
                        output = f"Ошибка аргументов инструмента: {error}"
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
            raise RuntimeError("Достигнут лимит: 100 шагов агента")


if __name__ == "__main__":
    if sys.argv[1:] == ["--mcp-server"]:
        mcp.run(transport="stdio") # нахер они эту mcp добавили..
    else:
        asyncio.run(run_agent(Path(sys.argv[1] if len(sys.argv) > 1 else "task.md")))
