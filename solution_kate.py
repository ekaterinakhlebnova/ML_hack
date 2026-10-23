import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from openai import AsyncOpenAI


def read_task(path: Path) -> str:
    return path.read_text(encoding="utf-8")


START_TIME = time.time()
TOTAL_LIMIT = 10800         
DEADLINE = START_TIME + TOTAL_LIMIT - 900  


def seconds_left() -> float:
    return DEADLINE - time.time()


def run_python(code: str) -> str:
    try:
        result = subprocess.run(
            [sys.executable, "-u", "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(30, min(900, int(seconds_left()))),
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


async def run_agent(task_path: Path, best_score=None, metric=None):
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
        8)Один вызов run_python ограничен 900 секундами. Проверяй гипотезы на подвыборке и малом числе итераций, полное обучение делай только для записи submission.csv.
        9)Сохраняй полезный код в .py-файлы в текущем каталоге и импортируй их в следующих шагах.
    """)
    if best_score is not None:
        systemPrompt += (
            f"\n        10)В текущем каталоге УЖЕ лежит submission.csv от базового решения. "
            f"Его результат на валидации: {metric} = {best_score:.5f}. "
            f"Перезаписывай submission.csv ТОЛЬКО если твоя кросс-валидация по той же метрике строго лучше этого числа. "
            f"Если не удалось улучшить, оставь файл как есть и заверши работу.\n"
        )

    messages = [
        {"role": "system", "content": systemPrompt},
        {"role": "user", "content": task},
    ]

    client = AsyncOpenAI()

    for step in range(100):
        if seconds_left() < 120:
            print(f"\n[Агент] время вышло, осталось {seconds_left():.0f} с, завершаю.")
            return
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

                output = await asyncio.to_thread(tools_mapping[func_name], **args)


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
    task_file = Path(sys.argv[1] if len(sys.argv) > 1 else "task.md").resolve()
    os.chdir(task_file.parent)


    best_score, metric = None, None
    try:
        import model
        task_info = model.build_task(task_file)
        metric = task_info["metric"]
        budget = int(os.getenv("SOLVE_TIME_BUDGET", "3600"))
        history = model.solve(task_info, task_file.parent / "submission.csv", time_budget=budget)
        kept = [h["score"] for h in history if h.get("kept")]
        best_score = kept[-1] if kept else None
        print(f"[solve] лучший скор на валидации: {metric} = {best_score}")
    except Exception as e:
        print(f"[solve] упал, Qwen начинает с нуля: {e!r}")


    if not os.getenv("OPENAI_API_KEY"):
        print("[Агент] OPENAI_API_KEY не задан, цикл Qwen пропущен.")
    else:
        try:
            asyncio.run(run_agent(task_file, best_score=best_score, metric=metric))
        except Exception as e:
            print(f"[Агент] завершился с ошибкой, submission.csv остаётся от этапа 1: {e!r}")
