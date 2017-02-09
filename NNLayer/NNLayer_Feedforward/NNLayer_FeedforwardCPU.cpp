//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A������
// CPU�����p
//======================================
#include"stdafx.h"

#include"NNLayer_Feedforward.h"
#include"NNLayer_FeedforwardBase.h"


class NNLayer_FeedforwardCPU : public NNLayer_FeedforwardBase
{
private:
	std::vector<std::vector<NEURON_TYPE>>	lppNeuron;			/**< �e�j���[�����̌W��<�j���[������, ���͐�> */
	std::vector<NEURON_TYPE>				lpBias;				/**< �j���[�����̃o�C�A�X<�j���[������> */
	std::vector<float>						lpOutput;			/**< �o�̓o�b�t�@ */
	std::vector<float>						lpDInputBuffer;		/**< ���͌덷����<���͐M����> */
	std::vector<float>						lpDOutputBuffer;	/**< �o�͌덷����<�j���[������> */

	float learnCoeff;	/**< �w�K�W�� */

	// Get�֐����g���Ə����s�������ނ̂ňꎞ�ۑ��p. PreCalculate�Œl���i�[.
	unsigned int inputBufferCount;				/**< ���̓o�b�t�@�� */
	unsigned int neuronCount;					/**< �j���[������ */
	std::vector<int> lpDOutputBufferPosition;	/**< �e�o�͐惌�C���[���ł̏o�͍����o�b�t�@�̈ʒu */

public:
	/** �R���X�g���N�^ */
	NNLayer_FeedforwardCPU(GUID guid)
		:	NNLayer_FeedforwardBase(guid)
		,	learnCoeff	(0.01f)
	{
	}
	/** �f�X�g���N�^ */
	virtual ~NNLayer_FeedforwardCPU()
	{
	}

public:
	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	ELayerKind GetLayerKind()const
	{
		return LAYER_KIND_CPU_CALC;
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ELayerErrorCode Initialize(void)
	{
		// ���̓o�b�t�@�����m�F
		unsigned int inputBufferCount = this->GetInputBufferCount();
		if(inputBufferCount == 0)
			return LAYER_ERROR_COMMON_OUT_OF_VALUERANGE;

		// �j���[���������m�F
		unsigned int neuronCount = this->GetNeuronCount();
		if(neuronCount == 0)
			return LAYER_ERROR_COMMON_OUT_OF_VALUERANGE;

		// �o�b�t�@���m�ۂ��A�����l��ݒ�
		this->lppNeuron.resize(neuronCount);
		this->lpBias.resize(neuronCount);
		for(unsigned int neuronNum=0; neuronNum<lppNeuron.size(); neuronNum++)
		{
			float maxArea = sqrt(6.0f / (0.5f*inputBufferCount + 0.5f*neuronCount));

			// �o�C�A�X
			this->lpBias[neuronNum] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f * maxArea;

			// �j���[����
			lppNeuron[neuronNum].resize(inputBufferCount);
			for(unsigned int inputNum=0; inputNum<lppNeuron[neuronNum].size(); inputNum++)
			{
				lppNeuron[neuronNum][inputNum] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f * maxArea;
			}
		}

		// �o�̓o�b�t�@���m��
		this->lpOutput.resize(neuronCount);
		// �o�͍����o�b�t�@�̊m��
		this->lpDInputBuffer.resize(inputBufferCount);

		return LAYER_ERROR_NONE;
	}
	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ELayerErrorCode Initialize(const INNLayerConfig& config)
	{
		// �ݒ����ݒ�
		ELayerErrorCode err = this->SetLayerConfig(config);
		if(err != LAYER_ERROR_NONE)
			return err;

		// ���ʂ̏���������
		return this->Initialize();
	}

	/** ���C���[���o�b�t�@�ɏ�������.
		@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
		@return ���������ꍇ0 */
	int WriteToBuffer(BYTE* o_lpBuffer)const
	{
		if(this->pConfig == NULL)
			return LAYER_ERROR_NONREGIST_CONFIG;

		int writeBufferByte = 0;

		// �ݒ���
		writeBufferByte += this->pConfig->WriteToBuffer(&o_lpBuffer[writeBufferByte]);

		// �j���[�����W��
		for(unsigned int neuronNum=0; neuronNum<this->lppNeuron.size(); neuronNum++)
		{
			memcpy(&o_lpBuffer[writeBufferByte], &this->lppNeuron[neuronNum][0], this->lppNeuron[neuronNum].size() * sizeof(NEURON_TYPE));
			writeBufferByte += this->lppNeuron[neuronNum].size() * sizeof(NEURON_TYPE);
		}

		// �o�C�A�X
		memcpy(&o_lpBuffer[writeBufferByte], &this->lpBias[0], this->lpBias.size() * sizeof(NEURON_TYPE));
		writeBufferByte += this->lpBias.size() * sizeof(NEURON_TYPE);

		return writeBufferByte;
	}
	/** ���C���[��ǂݍ���.
		@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
		@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
		@return	���ۂɓǂݎ�����o�b�t�@�T�C�Y. ���s�����ꍇ�͕��̒l */
	int ReadFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize)
	{
		int readBufferByte = 0;

		// �ݒ����ǂݍ���
		if(this->pConfig != NULL)
			delete this->pConfig;
		this->pConfig = ::CreateLayerConfigFromBuffer(i_lpBuffer, i_bufferSize, readBufferByte);

		// ����������
		this->Initialize();

		// �j���[�����W��
		for(unsigned int neuronNum=0; neuronNum<this->lppNeuron.size(); neuronNum++)
		{
			memcpy(&this->lppNeuron[neuronNum][0], &i_lpBuffer[readBufferByte], this->lppNeuron[neuronNum].size() * sizeof(NEURON_TYPE));
			readBufferByte += this->lppNeuron[neuronNum].size() * sizeof(NEURON_TYPE);
		}

		// �o�C�A�X
		memcpy(&this->lpBias[0], &i_lpBuffer[readBufferByte], this->lpBias.size() * sizeof(NEURON_TYPE));
		readBufferByte += this->lpBias.size() * sizeof(NEURON_TYPE);


		return readBufferByte;
	}

public:
	//================================
	// ���Z����
	//================================
	/** ���Z�O���������s����.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ELayerErrorCode PreCalculate()
	{
		// ���̓o�b�t�@�����m�F
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return LAYER_ERROR_FRAUD_INPUT_COUNT;

		// �j���[���������m�F
		this->neuronCount = this->GetNeuronCount();
		if(this->neuronCount == 0)
			return LAYER_ERROR_FRAUD_OUTPUT_COUNT;

		// �j���[�����o�b�t�@�̃T�C�Y�m�F
		if(this->lppNeuron.size() != this->neuronCount)
			return LAYER_ERROR_FRAUD_NEURON_COUNT;
		if(this->lppNeuron[0].size() != this->inputBufferCount)
			return LAYER_ERROR_FRAUD_NEURON_COUNT;

		// �o�̓o�b�t�@�̃T�C�Y���m�F
		if(this->lpOutput.size() != this->neuronCount)
			return LAYER_ERROR_FRAUD_OUTPUT_COUNT;

		// ���͍����o�b�t�@�̃T�C�Y���m�F
		if(this->lpDInputBuffer.size() != this->inputBufferCount)
			return LAYER_ERROR_FRAUD_OUTPUT_COUNT;

		// �o�͍����o�b�t�@���m��
		this->lpDOutputBuffer.resize(this->neuronCount);

		// �e�o�͐惌�C���[���ł̏o�͍����o�b�t�@�̈ʒu
		this->lpDOutputBufferPosition.resize(this->GetOutputToLayerCount());
		for(unsigned int layerNum=0; layerNum<this->GetOutputToLayerCount(); layerNum++)
		{
			this->lpDOutputBufferPosition[layerNum] = this->GetOutputToLayerByNum(layerNum)->GetInputBufferPositionByLayer(this);
		}

		return LAYER_ERROR_NONE;
	}

	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ELayerErrorCode Calculate()
	{
		for(unsigned int neuronNum=0; neuronNum<this->lppNeuron.size(); neuronNum++)
		{
			float tmp = 0;
			// �j���[�����̒l�����Z
			unsigned int inputNum = 0;
			for(unsigned int inputLayerNum=0; inputLayerNum<this->GetInputFromLayerCount(); inputLayerNum++)
			{
				auto pInputLayer = this->GetInputFromLayerByNum(inputLayerNum);
				for(unsigned int layerInputNum=0; layerInputNum<pInputLayer->GetOutputBufferCount(); layerInputNum++)
				{
					tmp += this->lppNeuron[neuronNum][inputNum] * pInputLayer->GetOutputBuffer()[layerInputNum];
					inputNum++;
				}
				// �o�C�A�X�����Z
				tmp += this->lpBias[neuronNum];
			}

			// �V�O���C�h�֐������Z
			this->lpOutput[neuronNum] = 1 / (1+exp(-tmp));
		}

		return LAYER_ERROR_NONE;
	}

	/** �o�̓f�[�^�o�b�t�@���擾����.
		�z��̗v�f����GetOutputBufferCount�̖߂�l.
		@return �o�̓f�[�^�z��̐擪�|�C���^ */
	const float* GetOutputBuffer()const
	{
		return &this->lpOutput[0];
	}
	/** �o�̓f�[�^�o�b�t�@���擾����.
		@param lpOutputBuffer	�o�̓f�[�^�i�[��z��. GetOutputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0 */
	ELayerErrorCode GetOutputBuffer(float lpOutputBuffer[])const
	{
		memcpy(lpOutputBuffer, &this->lpOutput[0], this->lpOutput.size() * sizeof(float));

		return LAYER_ERROR_NONE;
	}

public:
	//================================
	// �w�K����
	//================================
	/** �w�K�덷���v�Z����.
		���O�̌v�Z���ʂ��g�p���� */
	ELayerErrorCode CalculateLearnError()
	{
		// �o�͌덷�������v�Z
		for(unsigned int neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
		{
			// �e�o�͐惌�C���[�̏o�͍����̍��v�l�����߂�
			float sumDOutput = 0.0f;
			for(unsigned int outputToLayerNum=0; outputToLayerNum<this->GetOutputToLayerCount(); outputToLayerNum++)
			{
				sumDOutput += this->GetOutputToLayerByNum(outputToLayerNum)->GetDInputBuffer()[this->lpDOutputBufferPosition[outputToLayerNum] + neuronNum];
			}
			this->lpDOutputBuffer[neuronNum] = sumDOutput;
		}

		// ���͌덷�������v�Z
		unsigned int inputNum = 0;
		for(unsigned int inputLayerNum=0; inputLayerNum<this->GetInputFromLayerCount(); inputLayerNum++)
		{
			auto pInputLayer = this->GetInputFromLayerByNum(inputLayerNum);
			for(unsigned int layerInputNum=0; layerInputNum<pInputLayer->GetOutputBufferCount(); layerInputNum++)
			{
				float tmp = 0.0f;

				for(unsigned int neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
				{
					tmp += this->lpDOutputBuffer[neuronNum] * this->lppNeuron[neuronNum][inputNum];
				}

				this->lpDInputBuffer[inputNum] = pInputLayer->GetOutputBuffer()[layerInputNum] * (1.0f - pInputLayer->GetOutputBuffer()[layerInputNum]) * tmp;
				inputNum++;
			}
		}

		return LAYER_ERROR_NONE;
	}

	/** �덷���������C���[�ɔ��f������ */
	ELayerErrorCode ReflectionLearnError(void)
	{
		for(unsigned int neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
		{
			// �o�C�A�X�X�V
			this->lpBias[neuronNum] += this->learnCoeff * this->lpDOutputBuffer[neuronNum];

			// ���͑Ή��j���[�����X�V
			unsigned int inputNum = 0;
			for(unsigned int inputLayerNum=0; inputLayerNum<this->GetInputFromLayerCount(); inputLayerNum++)
			{
				auto pInputLayer = this->GetInputFromLayerByNum(inputLayerNum);
				for(unsigned int layerInputNum=0; layerInputNum<pInputLayer->GetOutputBufferCount(); layerInputNum++)
				{
					this->lppNeuron[neuronNum][inputNum] += this->learnCoeff * this->lpDOutputBuffer[neuronNum] * pInputLayer->GetOutputBuffer()[layerInputNum];
					inputNum++;
				}
			}
		}

		return LAYER_ERROR_NONE;
	}

	/** �w�K�������擾����.
		�z��̗v�f����GetOutputBufferCount�̖߂�l.
		@return	�덷�����z��̐擪�|�C���^ */
	const float* GetDInputBuffer()const
	{
		return &this->lpDInputBuffer[0];
	}
	/** �w�K�������擾����.
		@param lpDOutputBuffer	�w�K�������i�[����z��. GetOutputBufferCount�Ŏ擾�����l�̗v�f�����K�v */
	ELayerErrorCode GetDInputBuffer(float o_lpDInputBuffer[])const
	{
		memcpy(o_lpDInputBuffer, &this->lpDInputBuffer[0], this->lpDInputBuffer.size() * sizeof(float));

		return LAYER_ERROR_NONE;
	}
};


/** CPU�����p�̃��C���[���쐬 */
EXPORT_API CustomDeepNNLibrary::INNLayer* CreateLayerCPU(GUID guid)
{
	return new NNLayer_FeedforwardCPU(guid);

	return NULL;
}