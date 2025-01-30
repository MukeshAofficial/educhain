VISUAL_QUESTION_PROMPT_TEMPLATE = """Generate exactly {num} quantitative questions based on the topic: {topic}.
        Each question should require a visual representation of the data (bar graph, pie chart, line graph, or scatter plot or table) along with a detailed instruction on how to create that visual and options for the question. The question should be solvable based on the data in the visual.

        The visual type should be chosen based on the topic.
        Here is the general guidance for the visualization type:
        - Use pie chart when visualizing proportions or parts of a whole.
        - Use bar or column chart for comparing discrete categories or for displaying the frequency distribution.
        - Use line graph for displaying changes over time or continuous data or relationship between two continuous variables.
        - Use scatter plot for showing the relationship between two continuous variables, to identify any patterns and cluster of data.
        - Use table when presenting exact numerical data in organized rows and columns.

        The graph instruction MUST have the following structure in JSON format, selecting the relevant keys based on the visual type:
        {{
            "type": "bar" or "pie" or "line" or "scatter" or "table",
            "x_labels": ["label 1", "label 2", "label 3", "label 4"] for bar or line graphs,
            "x_values": [value 1, value 2, value 3, value 4] for scatter plot,
            "y_values": [value 1, value 2, value 3, value 4] for bar or line graphs,
            "labels": ["label 1", "label 2", "label 3", "label 4"] for pie chart,
            "sizes": [value 1, value 2, value 3, value 4] for pie chart,
            "y_label": "label for the y axis" for bar, line, scatter,
            "title": "title of the graph or table",
           "labels" : [ "label 1", "label 2", "label 3" ] for multiple lines in line graphs,
           "data": [
                    {{ "column1": "value1", "column2": "value2", ... }},
                    {{ "column1": "value3", "column2": "value4", ... }},
                    ...
                   ] for table
        }}

        Output the response in JSON format with the following structure:
        {{
          "questions" : [
            {{
                "question": "question text",
                "options": ["option a","option b", "option c", "option d"],
                "graph_instruction": {{"type": "bar" or "pie" or "line" or "scatter" or "table", ...}},
                "answer": "Correct answer of the question",
                "explanation": "Explanation of the question"
            }},
              {{
                "question": "question text",
                "options": ["option a","option b", "option c", "option d"],
                "graph_instruction": {{"type": "bar" or "pie" or "line" or "scatter" or "table", ...}},
                "answer": "Correct answer of the question",
                "explanation": "Explanation of the question"
            }}
           ]
        }}
"""
