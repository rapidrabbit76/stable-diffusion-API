guidance_scale = """
            조건부 신호(prompt)의 반영 강도. \n
            큰값을 사용하면 이미지가 좋아 보일 수 있지만 다양성이 떨어짐 \n
            일반적으로 7~8.5 값을 사용하는게 stable diffusion 에서는 안정적인 결과물을 생성
            """

prompt = """text prompt"""

init_image = """생성을 위한 기본 이미지"""

strength = """ 이미지에 추가되는 노이즈의 양. \n
               높을 수록 다양한 변형을 만들어 낼 수 있지만 \n
               조건으로 입력한 이미지 형태의 따르지 않음"""
