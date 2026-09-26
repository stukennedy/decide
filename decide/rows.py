"""Request shapes. Pure functions, no model imports, so the CLI stays fast and testable."""


def parse_option(text):
    """'billing' -> id billing; 'billing=Payments and charges' -> id + description."""
    oid, sep, description = text.partition("=")
    oid = oid.strip()
    if not oid:
        raise ValueError(f"empty option id in {text!r}")
    return {"id": oid, "description": description.strip() if sep and description.strip() else oid}


def ask_row(state, question, options):
    if not state.strip():
        raise ValueError("state is empty")
    if len(options) < 2:
        raise ValueError("give at least two options")
    parsed = [parse_option(o) for o in options]
    ids = [o["id"] for o in parsed]
    if len(set(ids)) != len(ids):
        raise ValueError("option ids must be unique")
    return {"id": "q", "state": state, "question": question, "options": parsed}


def ranked(result):
    """[(option_id, probability)] best first."""
    return sorted(zip(result["option_ids"], result["probabilities"]), key=lambda x: -x[1])


def jev_to_rows(request):
    """Jev-style {"state", "questions"} -> [(question_id, kind, row)].

    Options are mapped the way JevBench's semif_direct adapter maps them, so answers
    match the benchmarked configuration.
    """
    state, questions = request.get("state"), request.get("questions")
    if state is None:
        raise ValueError("state is required")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("questions must be a non-empty object")
    rows = []
    for qid, q in questions.items():
        if not isinstance(q, dict):
            raise ValueError(f"question {qid}: must be an object")
        kind, crit = q.get("type"), q.get("criteria")
        if kind == "noul":
            if crit is not None and not isinstance(crit, dict):
                raise ValueError(f"question {qid}: noul criteria must be an object")
            options = [{"id": k, "description": (crit or {}).get(k, f"The proposition is {k}.")}
                       for k in ("true", "false")]
        elif kind == "choice":
            if not isinstance(crit, dict) or not crit:
                raise ValueError(f"question {qid}: choice needs a criteria object")
            options = [{"id": k, "description": v or k} for k, v in crit.items()]
        elif kind == "score":
            if not isinstance(crit, list) or not 2 <= len(crit) <= 10:
                raise ValueError(f"question {qid}: score needs 2-10 levels")
            options = [{"id": str(i), "description": level} for i, level in enumerate(crit)]
        else:
            raise ValueError(f"question {qid}: type must be noul, choice or score")
        for o in options:
            o["description"] = f'{o["id"]}: {o["description"]}'
        rows.append((qid, kind, {"id": qid, "state": state, "question": q.get("instructions", ""),
                                 "options": options}))
    return rows


def jev_answer(kind, result):
    probs = dict(zip(result["option_ids"], result["probabilities"]))
    if kind == "noul":
        return {"type": "noul", "noul": probs["true"]}
    if kind == "choice":
        return {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs}
    return {"type": "score", "score": sum(int(k) * p for k, p in probs.items()), "probabilities": probs}
