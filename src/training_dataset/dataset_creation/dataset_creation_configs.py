triangle_prompt_template = """There is {a_or_no} triangle {triangle} in the figure. True or False?

A triangle is a polygon with three edges and three vertices, which are explicitly connected in the figure."""


quadrilateral_prompt_template = """There is {a_or_no} quadrilateral {quadrilateral} in this figure. True or False?

A quadrilateral is a four-sided polygon having four edges (sides) and four corners (vertices)."""


diameter_prompt_template = """In the figure, the line {line} is {a_or_not_a} diameter of a circle. True or False?"""


length_prompt_template = """Line {line1} is X times longer than {line2}. Which of the following options is a reasonable estimate of X? You only need to estimate from the visual information and do not need to do any mathematical reasoning. (a) 0.25 (b) 0.5 (c) 1 (d) 2 (e) 4"""


angle_prompt_template = """Which of the following options is a reasonable estimate of the angle {angle} in the figure? You only need to estimate from the visual information and do not need to do any mathematical reasoning. (a) 10 degrees (b) 45 degrees (c) 90 degrees (d) 135 degrees (e) 180 degrees"""


area_prompt_template = """{shape1} is X times larger in area than {shape2}. Which of the following options is a reasonable estimate? You only need to estimate from the visual information and do not need to do any mathematical reasoning. (a) 0.25 (b) 0.5 (c) 1 (d) 2 (e) 4"""
