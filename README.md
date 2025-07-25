-----

# 스픽뚝딱 (SpeakDddukddak)

스픽뚝딱은 **발음 교정 및 스피치 연습을 위한 AI 기반 웹 애플리케이션**입니다. 사용자의 음성을 분석하여 정확한 발음 피드백을 제공하고, 스피치 실력 향상을 위한 다양한 연습 시나리오를 제공합니다.

-----

# 주요 기능

  * **실시간 발음 분석**: 사용자의 음성을 실시간으로 분석하여 정확도, 유창성, 억양 등에 대한 피드백을 제공합니다.
  * **AI 기반 발음 교정**: 잘못된 발음 부분을 정확히 짚어내고, 개선을 위한 명확한 가이드를 제시합니다.
  * **다양한 스피치 연습 모드**:
      * **자유 연습 모드**: 원하는 텍스트를 입력하여 자유롭게 스피치 연습을 할 수 있습니다.
      * **상황별 시뮬레이션**: 면접, 발표, 일상 대화 등 다양한 실제 상황을 가정한 연습 시나리오를 제공합니다.
      * **스크립트 기반 연습**: 제공되는 스크립트를 따라 읽으며 발음과 억양을 연습할 수 있습니다.
  * **진행 상황 추적 및 리포트**: 사용자의 연습 기록을 저장하고, 시간이 지남에 따른 실력 향상도를 시각적으로 보여줍니다.
  * **커스터마이징 가능한 피드백**: 사용자가 원하는 피드백 항목을 선택하여 맞춤형 학습이 가능합니다.

-----

# 기술 스택

  * **Frontend**: React, Next.js
  * **Backend**: Python, FastAPI
  * **AI/ML**: TensorFlow/PyTorch (음성 인식 및 분석 모델)
  * **Database**: PostgreSQL
  * **Deployment**: Docker, AWS (예: EC2, S3, RDS)

-----

# 설치 및 실행 방법

1.  **리포지토리 클론**:

    ```bash
    git clone https://github.com/your-username/speakdddukddak.git
    cd speakdddukddak
    ```

2.  **환경 설정**:
    프로젝트 루트에 `.env` 파일을 생성하고 필요한 환경 변수를 설정합니다. (예: API 키, 데이터베이스 연결 정보 등)

    ```
    # .env 예시
    DATABASE_URL="postgresql://user:password@host:port/database"
    # 기타 필요한 환경 변수
    ```

3.  **의존성 설치**:

      * **Backend (Python)**:
        ```bash
        pip install -r requirements.txt
        ```
      * **Frontend (Node.js)**:
        ```bash
        npm install
        # 또는 yarn install
        ```

4.  **애플리케이션 실행**:

      * **Backend**:
        ```bash
        uvicorn main:app --reload
        ```
      * **Frontend**:
        ```bash
        npm run dev
        # 또는 yarn dev
        ```

5.  **접속**:
    브라우저에서 `http://localhost:3000` (프론트엔드) 에 접속하여 스픽뚝딱을 시작합니다. 백엔드 API는 `http://localhost:8000` 에서 실행됩니다.

