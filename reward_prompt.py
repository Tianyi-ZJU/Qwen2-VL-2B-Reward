def reward_prompt(query, response, score_range=(-5, 5)):
    """
    生成用于奖励模型训练的prompt
    
    参数:
        query (str): 用户提问
        response (str): 模型回答
        score_range (tuple): 打分范围，默认为(-5, 5)
        
    返回:
        str: 格式化的prompt
    """
    min_score, max_score = score_range
    
    prompt_str = f"""Please rate the quality of the following answer based on the image content. Your rating should consider accuracy, completeness, relevance, and clarity.
A high-quality answer should:
1) Accurately describe the image content
2) Completely address all aspects of the question
3) Be directly relevant to the question
4) Be clearly articulated and easy to understand

Question: {query}
Answer: {response}

Please rate this answer on a scale from {min_score} to {max_score}, where {min_score} is the worst possible quality and {max_score} is the best possible quality."""
    
    return prompt_str 