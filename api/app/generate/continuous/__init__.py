"""Continuous-mode generators (PLAN-continuous §6, Track G).

Modules (each ``render(spec: MotionSpec) -> str``, pure, number-provenance checked):
``technical``, ``llm_prompt``, ``css``, ``js``. Dispatched by
``app.generate.render_all`` via ``CONTINUOUS_GENERATOR_MODULES`` when ``spec.mode ==
"continuous"``. Until Track G lands they are stubs raising ``NotImplementedError``.
"""
