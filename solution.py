import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

from openai import AsyncOpenAI


def read_task(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def run_python(code: str) -> str:
    """Выполнить Python-код и вернуть stdout, stderr и код возврата.

    Каждый вызов — новый процесс. Файлы сохраняются, переменные — нет.
    Рабочий каталог совпадает с каталогом Markdown-файла задачи.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-u", "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",  # Явно указываем UTF-8
            errors="replace",  # Заменяем нечитаемые символы вместо падения
            timeout=900,
            env={"PATH": os.defpath, "PYTHONIOENCODING": "utf-8"},
        )

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        return json.dumps({
            "stdout": stdout[-12000:],
            "stderr": stderr[-12000:],
            "exit_code": result.returncode
        }, ensure_ascii=False)
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Превышен лимит выполнения: 900 секунд"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Ошибка выполнения процесса: {str(e)}"}, ensure_ascii=False)


# Описание инструментов для OpenAI API
tools = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Выполнить Python-код и вернуть stdout, stderr и код возврата.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python код для выполнения в терминале."
                    }
                },
                "required": ["code"]
            }
        }
    }
]

tools_mapping = {
    "run_python": run_python
}


async def run_agent(task_path: Path):
    task_path = task_path.resolve()
    task = read_task(task_path)
    model = os.getenv("OPENAI_MODEL", "qwen/qwen3.8-27b")
    systemPrompt = ("""
        Ты - профессиональный автономный Data Science агент. У тебя нет доступа к интернету.
        Ты имеешь доступ исключительно к каталогу в котором находится файл solution.py.
        Все зависимости, которые могли быть использованы описаны в файле requirements.txt.
        Твоя задача - на основе задачи, которая будет подана пользователем построить ML-модель.
        При этом тебе нужно самостоятельно по данным, которые будут предоставлены в .csv файлах определить тип задачи.
        Для определения типа задачи можешь использовать библиотеку matplotlib. Она имеется в виртуальном окружнии контейнера.
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

    client = AsyncOpenAI()

    for step in range(100):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                parallel_tool_calls=False,
            )
        except Exception as e:
            print(f"Ошибка при вызове OpenAI API: {e}")
            await asyncio.sleep(5)
            continue

        message = response.choices[0].message

        # Если модель решила написать мысли словами (редко при вызове тулз)
        if message.content:
            print(f"\n[Мысли LLM]:\n{message.content}")

        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            print("\n[Агент завершил работу! Текст ответа]:")
            print(message.content or "Работа завершена.")
            return

        for call in message.tool_calls:
            func_name = call.function.name
            try:
                args = json.loads(call.function.arguments)
            except json.JSONDecodeError as err:
                output = f"Ошибка декодирования аргументов JSON: {err}"
                messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
                continue

            if func_name in tools_mapping:
                code_to_run = args.get('code', '')
                print(f"\n{'=' * 20} [Шаг {step + 1}] МОДЕЛЬ НАПИСАЛА КОД {'=' * 20}")
                print(code_to_run)
                print(f"{'=' * 60}\n")

                # Запускаем код
                output = await asyncio.to_thread(tools_mapping[func_name], **args)

                # Красиво выводим результат работы кода
                try:
                    res_dict = json.loads(output)
                    print(f"--- [Результат выполнения | Exit Code: {res_dict.get('exit_code')}] ---")
                    if res_dict.get('stdout'):
                        print("STDOUT:\n" + res_dict['stdout'])
                    if res_dict.get('stderr'):
                        print("STDERR:\n" + res_dict['stderr'])
                    if res_dict.get('error'):
                        print("ERROR:\n" + res_dict['error'])
                except:
                    print(f"--- [Результат выполнения] ---\n{output}")
                print("-" * 60 + "\n")
            else:
                output = f"Инструмент {func_name} не найден."
                print(output)

            messages.append({"role": "tool", "tool_call_id": call.id, "content": output})

    raise RuntimeError("Достигнут лимит: 100 шагов агента")


if __name__ == "__main__":
    task_file = Path(sys.argv[1] if len(sys.argv) > 1 else "task.md")
    # Переходим в рабочую директорию папки с задачей, чтобы относительные пути работали правильно
    os.chdir(task_file.parent)
    asyncio.run(run_agent(task_file))

