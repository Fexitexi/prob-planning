import clingo



def on_model(m):
    print(m.symbols(shown=True))

ctl = clingo.Control()
lines = []
lines2 = []
lines.append("a(X) :- X = 30/7.")
lines2.append("b.")
ctl.add("base2", [], "\n".join(lines2))
ctl.ground([("base", [])])
ctl.ground([("base2", [])])
ctl.ground([("base", [])])
ctl.ground([("base", [])])
ctl.solve(on_model=on_model)
ctl.ground([("base", [])])
ctl.ground([("base2", [])])
ctl.ground([("base", [])])
lines.append("b.")
ctl.add("base", [], "\n".join(lines))
ctl.ground([("base", [])])
ctl.solve(on_model=on_model)
