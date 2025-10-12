import onnx, numpy as np
from onnx import helper, numpy_helper

def expand_layernorm(model):
    g = model.graph
    new_nodes = []
    replaced = 0
    for node in g.node:
        if node.op_type != "LayerNormalization":
            new_nodes.append(node)
            continue

        x, gamma, beta = node.input[:3]
        eps = 1e-5
        for a in node.attribute:
            if a.name == "epsilon": eps = a.f

        pref = (node.name or "ln") + f"_{replaced}"

        mean = pref + "_mean"
        new_nodes.append(helper.make_node("ReduceMean", [x], [mean], keepdims=1, axes=[-1]))

        centered = pref + "_centered"
        new_nodes.append(helper.make_node("Sub", [x, mean], [centered]))

        sq = pref + "_sq"
        new_nodes.append(helper.make_node("Mul", [centered, centered], [sq]))

        var = pref + "_var"
        new_nodes.append(helper.make_node("ReduceMean", [sq], [var], keepdims=1, axes=[-1]))

        eps_name = pref + "_eps"
        g.initializer.append(numpy_helper.from_array(np.array([eps], np.float32), eps_name))

        var_eps = pref + "_var_eps"
        new_nodes.append(helper.make_node("Add", [var, eps_name], [var_eps]))

        denom = pref + "_denom"
        new_nodes.append(helper.make_node("Sqrt", [var_eps], [denom]))

        norm = pref + "_norm"
        new_nodes.append(helper.make_node("Div", [centered, denom], [norm]))

        scaled = pref + "_scaled"
        new_nodes.append(helper.make_node("Mul", [norm, gamma], [scaled]))

        new_nodes.append(helper.make_node("Add", [scaled, beta], node.output))

        replaced += 1

    g.ClearField("node")
    g.node.extend(new_nodes)
    return replaced

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python expand_layernorm.py in.onnx out.onnx")
        sys.exit(1)
    model = onnx.load(sys.argv[1])
    n = expand_layernorm(model)
    onnx.checker.check_model(model)
    onnx.save(model, sys.argv[2])
    print(f"✅ Expanded {n} LayerNormalization node(s). Saved to {sys.argv[2]}")
