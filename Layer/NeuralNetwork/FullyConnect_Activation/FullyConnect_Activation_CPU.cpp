//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A������
// CPU�����p
//======================================
#include"stdafx.h"

#include"FullyConnect_Activation_DATA.hpp"
#include"FullyConnect_Activation_FUNC.hpp"
#include"FullyConnect_Activation_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

class FeedforwardCPU : public FullyConnect_Activation_Base
{
private:
	// �{��
	std::vector<std::vector<NEURON_TYPE>>	lppNeuron;			/**< �e�j���[�����̌W��<�j���[������, ���͐�> */
	std::vector<NEURON_TYPE>				lpBias;				/**< �j���[�����̃o�C�A�X<�j���[������> */

	// ���o�̓o�b�t�@
	std::vector<std::vector<F32>>			lpOutputBuffer;		/**< �o�̓o�b�t�@ <�o�b�`��><�j���[������> */
	std::vector<std::vector<F32>>			lpDInputBuffer;		/**< ���͌덷���� <�o�b�`��><���͐M����> */
	
	std::vector<F32*>						lppBatchOutputBuffer;		/**< �o�b�`�����p�o�̓o�b�t�@ <�o�b�`��> */
	std::vector<F32*>						lppBatchDInputBuffer;		/**< �o�b�`�����p���͌덷���� <�o�b�`��> */

	// Get�֐����g���Ə����s�������ނ̂ňꎞ�ۑ��p. PreCalculate�Œl���i�[.
	U32 inputBufferCount;				/**< ���̓o�b�t�@�� */
	U32 neuronCount;					/**< �j���[������ */
	U32 outputBufferCount;				/**< �o�̓o�b�t�@�� */

	// ���Z���̓��̓f�[�^
	CONST_BATCH_BUFFER_POINTER m_lppInputBuffer;	/**< ���Z���̓��̓f�[�^ */
	CONST_BATCH_BUFFER_POINTER m_lppDOutputBuffer;	/**< ���͌덷�v�Z���̏o�͌덷�f�[�^ */

	// ���Z�����p�̃o�b�t�@
	bool onUseDropOut;											/**< �h���b�v�A�E�g���������s����t���O. */
	std::vector<std::vector<NEURON_TYPE>>	lppDropOutBuffer;	/**< �h���b�v�A�E�g�����p�̌W��<�j���[������, ���͐�> */

public:
	/** �R���X�g���N�^ */
	FeedforwardCPU(Gravisbell::GUID guid)
		:	FullyConnect_Activation_Base	(guid)
		,	inputBufferCount				(0)		/**< ���̓o�b�t�@�� */
		,	neuronCount						(0)		/**< �j���[������ */
		,	outputBufferCount				(0)		/**< �o�̓o�b�t�@�� */
		,	m_lppInputBuffer				(NULL)	/**< ���Z���̓��̓f�[�^ */
		,	m_lppDOutputBuffer				(NULL)	/**< ���͌덷�v�Z���̏o�͌덷�f�[�^ */
		,	onUseDropOut					(false)	
	{
	}
	/** �f�X�g���N�^ */
	virtual ~FeedforwardCPU()
	{
	}

public:
	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode Initialize(void)
	{
		// ���̓o�b�t�@�����m�F
		unsigned int inputBufferCount = this->GetInputBufferCount();
		if(inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_VALUERANGE;

		// �j���[���������m�F
		unsigned int neuronCount = this->GetNeuronCount();
		if(neuronCount == 0)
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_VALUERANGE;

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

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@param	i_config			�ݒ���
		@oaram	i_inputDataStruct	���̓f�[�^�\�����
		@return	���������ꍇ0 */
	ErrorCode Initialize(const SettingData::Standard::IData& i_data, const IODataStruct& i_inputDataStruct)
	{
		ErrorCode err;

		// �ݒ���̓o�^
		err = this->SetLayerConfig(i_data);
		if(err != ErrorCode::ERROR_CODE_NONE)
			return err;

		// ���̓f�[�^�\���̐ݒ�
		this->inputDataStruct = i_inputDataStruct;

		return this->Initialize();
	}
	/** ������. �o�b�t�@����f�[�^��ǂݍ���
		@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
		@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
		@return	���������ꍇ0 */
	ErrorCode InitializeFromBuffer(BYTE* i_lpBuffer, int i_bufferSize)
	{
		int readBufferByte = 0;

		// �ݒ����ǂݍ���
		SettingData::Standard::IData* pLayerStructure = CreateLayerStructureSettingFromBuffer(i_lpBuffer, i_bufferSize, readBufferByte);
		if(pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_INITLAYER_READ_CONFIG;
		this->SetLayerConfig(*pLayerStructure);
		delete pLayerStructure;

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


		return ErrorCode::ERROR_CODE_NONE;
	}

	//===========================
	// ���C���[�ۑ�
	//===========================
	/** ���C���[���o�b�t�@�ɏ�������.
		@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
		@return ���������ꍇ0 */
	int WriteToBuffer(BYTE* o_lpBuffer)const
	{
		if(this->pLayerStructure == NULL)
			return ErrorCode::ERROR_CODE_NONREGIST_CONFIG;

		int writeBufferByte = 0;

		// �ݒ���
		writeBufferByte += this->pLayerStructure->WriteToBuffer(&o_lpBuffer[writeBufferByte]);

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

public:
	//================================
	// ���Z����
	//================================
	/** ���Z�O���������s����.(�w�K�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
	ErrorCode PreProcessLearn(unsigned int batchSize)
	{
		ErrorCode errorCode = this->PreProcessCalculate(batchSize);
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// ���͍����o�b�t�@���쐬
		this->lpDInputBuffer.resize(this->batchSize);
		this->lppBatchDInputBuffer.resize(this->batchSize);
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			this->lpDInputBuffer[batchNum].resize(this->inputBufferCount);
			this->lppBatchDInputBuffer[batchNum] = &this->lpDInputBuffer[batchNum][0];
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode PreProcessCalculate(unsigned int batchSize)
	{
		this->batchSize = batchSize;

		// ���̓o�b�t�@�����m�F
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;
		
		// �j���[���������m�F
		this->neuronCount = this->GetNeuronCount();
		if(this->neuronCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;

		// �o�̓o�b�t�@�����m�F
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// �j���[�����o�b�t�@�̃T�C�Y�m�F
		if(this->lppNeuron.size() != this->neuronCount)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;
		if(this->lppNeuron[0].size() != this->inputBufferCount)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;


		// �o�̓o�b�t�@���쐬
		this->lpOutputBuffer.resize(this->batchSize);
		this->lppBatchOutputBuffer.resize(this->batchSize);
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			this->lpOutputBuffer[batchNum].resize(this->outputBufferCount);
			this->lppBatchOutputBuffer[batchNum] = &this->lpOutputBuffer[batchNum][0];
		}

		// �h���b�v�A�E�g�����𖢎g�p�ɕύX
		this->onUseDropOut = false;
		this->lppDropOutBuffer.clear();

		// ���͍����o�b�t�@���쐬
		// �̓X�L�b�v

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode PreProcessLearnLoop(const SettingData::Standard::IData& data)
	{
		if(this->pLearnData != NULL)
			delete this->pLearnData;
		this->pLearnData = data.Clone();

		// �h���b�v�A�E�g
		{
			auto pItem = dynamic_cast<const Gravisbell::SettingData::Standard::IItem_Float*>(data.GetItemByID(L"DropOut"));
			if(pItem)
				this->learnData.DropOut = pItem->GetValue();
			else
				this->learnData.DropOut = 0.0f;
			
			S32 dropOutRate = (S32)(learnData.DropOut * RAND_MAX);

			if(dropOutRate > 0)
			{
				this->onUseDropOut = true;
				if(this->lppDropOutBuffer.empty())
				{
					// �o�b�t�@�̊m��
					this->lppDropOutBuffer.resize(this->neuronCount);
					for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
						this->lppDropOutBuffer[neuronNum].resize(this->inputBufferCount);
				}

				// �o�b�t�@��1or0�����
				// 1 : DropOut���Ȃ�
				// 0 : DropOut����
				for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
				{
					for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
					{
						if(rand() < dropOutRate)	// �h���b�v�A�E�g����
							this->lppDropOutBuffer[neuronNum][inputNum] = 0.0f;
						else
							this->lppDropOutBuffer[neuronNum][inputNum] = 1.0f;
					}
				}
			}
			else
			{
				this->onUseDropOut = false;
				this->lppDropOutBuffer.clear();
			}
		}
		// �w�K�W��
		{
			auto pItem = dynamic_cast<const Gravisbell::SettingData::Standard::IItem_Float*>(data.GetItemByID(L"LearnCoeff"));
			if(pItem)
				this->learnData.LearnCoeff = pItem->GetValue();
			else
				this->learnData.LearnCoeff = 1.0f;
		}

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}
	/** ���Z���[�v�̏���������.�f�[�^�Z�b�g�̉��Z�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode PreProcessCalculateLoop()
	{
		this->onUseDropOut = false;

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer)
	{
		this->m_lppInputBuffer = i_lppInputBuffer;

		// ���[�v����if�������s����Ə����s�������ނ̂ŊO�ŕ�����.
		if(!this->onUseDropOut)
		{
			// DropOut���g�p���Ȃ��ꍇ
			for(unsigned int batchNum=0; batchNum<this->batchSize; batchNum++)
			{
				for(unsigned int neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
				{
					float tmp = 0;

					// �j���[�����̒l�����Z
					for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
					{
						tmp += i_lppInputBuffer[batchNum][inputNum] * this->lppNeuron[neuronNum][inputNum];
					}
					tmp += this->lpBias[neuronNum];

					if(this->learnData.DropOut > 0.0f)
						tmp *= (1.0f - this->learnData.DropOut);

					// �������֐�
					if(this->layerStructure.ActivationType == Gravisbell::Layer::NeuralNetwork::FullyConnect_Activation::LayerStructure::ActivationType_ReLU)
					{
						// ReLU
						this->lpOutputBuffer[batchNum][neuronNum] = max(0.0f, tmp);
					}
					else
					{
						// �V�O���C�h�֐������Z
						this->lpOutputBuffer[batchNum][neuronNum] = 1 / (1+exp(-tmp));
					}
					
					#ifdef _DEBUG
					if(isnan(this->lpOutputBuffer[batchNum][neuronNum]))
						return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
					#endif
				}
			}
		}
		else
		{
			// DropOut���g�p����ꍇ
			for(unsigned int batchNum=0; batchNum<this->batchSize; batchNum++)
			{
				for(unsigned int neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
				{
					float tmp = 0;

					// �j���[�����̒l�����Z
					for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
					{
						// ��DropOut�̗L���ňႤ�̂͂��̈�s����
						tmp += i_lppInputBuffer[batchNum][inputNum] * this->lppNeuron[neuronNum][inputNum] * this->lppDropOutBuffer[neuronNum][inputNum];
						
					#ifdef _DEBUG
					if(isnan(tmp))
						return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
					#endif
					}
					tmp += this->lpBias[neuronNum];

					// �������֐�
					if(this->layerStructure.ActivationType == Gravisbell::Layer::NeuralNetwork::FullyConnect_Activation::LayerStructure::ActivationType_ReLU)
					{
						// ReLU
						this->lpOutputBuffer[batchNum][neuronNum] = max(0.0f, tmp);
					}
					else
					{
						// �V�O���C�h�֐������Z
						this->lpOutputBuffer[batchNum][neuronNum] = 1 / (1+exp(-tmp));
					}

					#ifdef _DEBUG
					if(isnan(this->lpOutputBuffer[batchNum][neuronNum]))
						return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
					#endif
				}
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �o�̓f�[�^�o�b�t�@���擾����.
		�z��̗v�f����GetOutputBufferCount�̖߂�l.
		@return �o�̓f�[�^�z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const
	{
		return &this->lppBatchOutputBuffer[0];
	}
	/** �o�̓f�[�^�o�b�t�@���擾����.
		@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
		@return ���������ꍇ0 */
	ErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
	{
		if(o_lpOutputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 outputBufferCount = this->GetOutputBufferCount();

		for(U32 batchNum=0; batchNum<batchSize; batchNum++)
		{
			memcpy(o_lpOutputBuffer[batchNum], this->lppBatchOutputBuffer[batchNum], sizeof(F32)*outputBufferCount);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

public:
	//================================
	// �w�K����
	//================================
	/** �w�K�덷���v�Z����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode CalculateLearnError(CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		this->m_lppDOutputBuffer = i_lppDOutputBuffer;

		// ���[�v����if�������s����Ə����s�������ނ̂ŊO�ŕ�����.
		if(!this->onUseDropOut)
		{
			// DropOut����
			for(unsigned int batchNum=0; batchNum<this->batchSize; batchNum++)
			{
				// ���͌덷�������v�Z
				for(unsigned int inputNum=0; inputNum<this->inputBufferCount; inputNum++)
				{
					float tmp = 0.0f;

					for(unsigned int neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
					{
						tmp += i_lppDOutputBuffer[batchNum][neuronNum] * this->lppNeuron[neuronNum][inputNum];
					}
#ifdef _DEBUG
					if(isnan(tmp))
						return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
#endif

					// �������֐��ŕ���
					if(this->layerStructure.ActivationType == Gravisbell::Layer::NeuralNetwork::FullyConnect_Activation::LayerStructure::ActivationType_ReLU)
					{
						// ReLU
						this->lpDInputBuffer[batchNum][inputNum] = (float)(this->m_lppInputBuffer[batchNum][inputNum] > 0.0f) * tmp;
					}
					else
					{
						// �V�O���C�h
						this->lpDInputBuffer[batchNum][inputNum] = min(1.0f, this->m_lppInputBuffer[batchNum][inputNum]) * (1.0f - min(1.0f, this->m_lppInputBuffer[batchNum][inputNum])) * tmp;
					}
					#ifdef _DEBUG
						if(isnan(lpDInputBuffer[batchNum][inputNum]))
							return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
					#endif
				}
			}
		}
		else
		{
			// DropOut�L��
			for(unsigned int batchNum=0; batchNum<this->batchSize; batchNum++)
			{
				// ���͌덷�������v�Z
				for(unsigned int inputNum=0; inputNum<this->inputBufferCount; inputNum++)
				{
					float tmp = 0.0f;

					for(unsigned int neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
					{
						// DropOut�̗L���ňႤ�̂͂��̈ꕶ����
						tmp += i_lppDOutputBuffer[batchNum][neuronNum] * this->lppNeuron[neuronNum][inputNum] * this->lppDropOutBuffer[neuronNum][inputNum];

						#ifdef _DEBUG
						if(isnan(i_lppDOutputBuffer[batchNum][neuronNum]))
							return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
						if(isnan(this->lppNeuron[neuronNum][inputNum]))
							return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
						#endif
					}

					#ifdef _DEBUG
					if(isnan(tmp))
						return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
					#endif

					// �������֐��ŕ���
					if(this->layerStructure.ActivationType == Gravisbell::Layer::NeuralNetwork::FullyConnect_Activation::LayerStructure::ActivationType_ReLU)
					{
						// ReLU
						this->lpDInputBuffer[batchNum][inputNum] = (float)(1.0f * (this->m_lppInputBuffer[batchNum][inputNum] > 0.0f)) * tmp;
						#ifdef _DEBUG
						if(isnan(this->lpDInputBuffer[batchNum][inputNum]))
							return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
						#endif
					}
					else
					{
						// �V�O���C�h
						this->lpDInputBuffer[batchNum][inputNum] = min(1.0f, this->m_lppInputBuffer[batchNum][inputNum]) * (1.0f - min(1.0f, this->m_lppInputBuffer[batchNum][inputNum])) * tmp;
					}
					#ifdef _DEBUG
					if(isnan(lpDInputBuffer[batchNum][inputNum]))
						return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
					#endif
				}
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K���������C���[�ɔ��f������.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		�o�͌덷�����A���͌덷�����͒��O��CalculateLearnError�̒l���Q�Ƃ���. */
	ErrorCode ReflectionLearnError(void)
	{
		for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
		{
			// �o�C�A�X�X�V
			{
				F32 sumDOutput = 0.0f;
				for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
				{
					 sumDOutput += this->m_lppDOutputBuffer[batchNum][neuronNum];
				}
				
				#ifdef _DEBUG
				if(isnan(sumDOutput))
					return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
				#endif

				this->lpBias[neuronNum] += this->learnData.LearnCoeff * sumDOutput;// / this->batchSize;
			}

			// ���͑Ή��j���[�����X�V
			for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
			{
				F32 sumDOutput = 0.0f;
				for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
				{
					sumDOutput += this->m_lppDOutputBuffer[batchNum][neuronNum] * this->m_lppInputBuffer[batchNum][inputNum];
				}
								
				#ifdef _DEBUG
				if(isnan(sumDOutput))
					return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
				#endif

				this->lppNeuron[neuronNum][inputNum] += this->learnData.LearnCoeff * sumDOutput;// / this->batchSize;
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K�������擾����.
		�z��̗v�f����[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]
		@return	�덷�����z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER GetDInputBuffer()const
	{
		return &this->lppBatchDInputBuffer[0];
	}
	/** �w�K�������擾����.
		@param lpDInputBuffer	�w�K�������i�[����z��.[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̔z�񂪕K�v */
	ErrorCode GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
	{
		if(o_lpDInputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 inputBufferCount = this->GetInputBufferCount();

		for(U32 batchNum=0; batchNum<batchSize; batchNum++)
		{
			memcpy(o_lpDInputBuffer[batchNum], this->lppBatchDInputBuffer[batchNum], sizeof(F32)*inputBufferCount);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

};


/** CPU�����p�̃��C���[���쐬 */
EXPORT_API Gravisbell::Layer::NeuralNetwork::INNLayer* CreateLayerCPU(Gravisbell::GUID guid)
{
	return new FeedforwardCPU(guid);
}