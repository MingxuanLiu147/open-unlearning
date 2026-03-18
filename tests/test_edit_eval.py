from evals.edit import EditGeneralizationEvaluator


def test_normalize_rephrase_prompts_accepts_rephrase_alias():
    evaluator = EditGeneralizationEvaluator({})

    prompts = evaluator._normalize_rephrase_prompts(
        {"rephrase": "Leonardo DiCaprio's country of citizenship is known as"}
    )

    assert prompts == ["Leonardo DiCaprio's country of citizenship is known as"]
