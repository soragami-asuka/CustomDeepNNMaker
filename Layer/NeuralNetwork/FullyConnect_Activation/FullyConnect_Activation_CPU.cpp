//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A������
// CPU�����p
//======================================
#include"stdafx.h"

#include"FullyConnect_Activation_DATA.hpp"
#include"FullyConnect_Activation_FUNC.hpp"
#include"FullyConnect_Activation_Base.h"

#include"FullyConnect_Activation_CPU.h"
#include"FullyConnect_Activation_LayerData_CPU.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace
{
	//================================
	// �������֐�
	//================================
	// lenear�n
	F32 func_activation_lenear(F32 x)
	{
		return x;
	}
	F32 func_dactivation_lenear(F32 x)
	{
		return 1;
	}

	// sigmoid�n
	F32 func_activation_sigmoid(F32 x)
	{
		return 1.0f / (1.0f + (F32)exp(-x));
	}
	F32 func_dactivation_sigmoid(F32 x)
	{
		return x * (1.0f - x);
	}

	F32 func_activation_sigmoid_crossEntropy(F32 x)
	{
		return 1.0f / (1.0f + (F32)exp(-x));
	}
	F32 func_dactivation_sigmoid_crossEntropy(F32 x)
	{
		return 1.0f;
	}

	// ReLU�n
	F32 func_activation_ReLU(F32 x)
	{
		return x * (x > 0.0f);
	}
	F32 func_dactivation_ReLU(F32 x)
	{
		return 1.0f * (x > 0.0f);
	}

	// SoftMax�n
	F32 func_activation_SoftMax(F32 x)
	{
		return (F32)exp(x);	// ���ς͕ʂɍs��
	}


}


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** �R���X�g���N�^ */
	FullyConnect_Activation_CPU::FullyConnect_Activation_CPU(Gravisbell::GUID guid, FullyConnect_Activation_LayerData_CPU& i_layerData)
		:	FullyConnect_Activation_Base	(guid)
		,	layerData						(i_layerData)	/**< ���C���[�f�[�^ */
		,	inputBufferCount				(0)		/**< ���̓o�b�t�@�� */
		,	neuronCount						(0)		/**< �j���[������ */
		,	outputBufferCount				(0)		/**< �o�̓o�b�t�@�� */
		,	m_lppInputBuffer				(NULL)	/**< ���Z���̓��̓f�[�^ */
		,	m_lppDOutputBufferPrev			(NULL)	/**< ���͌덷�v�Z���̏o�͌덷�f�[�^ */
		,	onUseDropOut					(false)
		,	func_activation					(func_activation_sigmoid)
		,	func_dactivation				(func_dactivation_sigmoid)
	{
	}
	/** �f�X�g���N�^ */
	FullyConnect_Activation_CPU::~FullyConnect_Activation_CPU()
	{
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 FullyConnect_Activation_CPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode FullyConnect_Activation_CPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	FullyConnect_Activation_LayerData_Base& FullyConnect_Activation_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const FullyConnect_Activation_LayerData_Base& FullyConnect_Activation_CPU::GetLayerData()const
	{
		return this->layerData;
	}


	//===========================
	// ���C���[�ۑ�
	//===========================
	/** ���C���[���o�b�t�@�ɏ�������.
		@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
		@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
	S32 FullyConnect_Activation_CPU::WriteToBuffer(BYTE* o_lpBuffer)const
	{
		return this->layerData.WriteToBuffer(o_lpBuffer);
	}


	//================================
	// ���Z����
	//================================
	/** ���Z�O���������s����.(�w�K�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
	ErrorCode FullyConnect_Activation_CPU::PreProcessLearn(unsigned int batchSize)
	{
		ErrorCode errorCode = this->PreProcessCalculate(batchSize);
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;
		
		// �o�͌덷�o�b�t�@���쐬
		this->lpDOutputBuffer.resize(this->batchSize);
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			this->lpDOutputBuffer[batchNum].resize(this->outputBufferCount);
		}

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
	ErrorCode FullyConnect_Activation_CPU::PreProcessCalculate(unsigned int batchSize)
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
		if(this->layerData.lppNeuron.size() != this->neuronCount)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;
		if(this->layerData.lppNeuron[0].size() != this->inputBufferCount)
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

		// �������֐���ݒ�
		switch(this->layerData.layerStructure.ActivationType)
		{
			// lenear
		case Gravisbell::Layer::NeuralNetwork::FullyConnect_Activation::LayerStructure::ActivationType_lenear:
			this->func_activation  = ::func_activation_lenear;
			this->func_dactivation = ::func_dactivation_lenear;
			break;

			// Sigmoid
		case Gravisbell::Layer::NeuralNetwork::FullyConnect_Activation::LayerStructure::ActivationType_sigmoid:
		default:
			this->func_activation  = ::func_activation_sigmoid;
			this->func_dactivation = ::func_dactivation_sigmoid;
			break;
		case Gravisbell::Layer::NeuralNetwork::FullyConnect_Activation::LayerStructure::ActivationType_sigmoid_crossEntropy:
			this->func_activation  = ::func_activation_sigmoid_crossEntropy;
			this->func_dactivation = ::func_dactivation_sigmoid_crossEntropy;
			break;

			// ReLU
		case Gravisbell::Layer::NeuralNetwork::FullyConnect_Activation::LayerStructure::ActivationType_ReLU:
			this->func_activation  = ::func_activation_ReLU;
			this->func_dactivation = ::func_dactivation_ReLU;
			break;

			// Softmax
		case Gravisbell::Layer::NeuralNetwork::FullyConnect_Activation::LayerStructure::ActivationType_softmax:
			this->func_activation  = ::func_activation_SoftMax;
			this->func_dactivation = ::func_dactivation_sigmoid;
			break;
		case Gravisbell::Layer::NeuralNetwork::FullyConnect_Activation::LayerStructure::ActivationType_softmax_crossEntropy:
			this->func_activation  = ::func_activation_SoftMax;
			this->func_dactivation = ::func_dactivation_sigmoid_crossEntropy;
			break;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode FullyConnect_Activation_CPU::PreProcessLearnLoop(const SettingData::Standard::IData& data)
	{
		if(this->pLearnData != NULL)
			delete this->pLearnData;
		this->pLearnData = data.Clone();

		// �h���b�v�A�E�g
		{
			S32 dropOutRate = (S32)(this->layerData.layerStructure.DropOut * RAND_MAX);

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
	ErrorCode FullyConnect_Activation_CPU::PreProcessCalculateLoop()
	{
		this->onUseDropOut = false;

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode FullyConnect_Activation_CPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer)
	{
		this->m_lppInputBuffer = i_lppInputBuffer;

		for(unsigned int batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			for(unsigned int neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
			{
				float tmp = 0;

				// �j���[�����̒l�����Z
				for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
				{
					if(this->onUseDropOut)
						tmp += i_lppInputBuffer[batchNum][inputNum] * this->layerData.lppNeuron[neuronNum][inputNum] * this->lppDropOutBuffer[neuronNum][inputNum];
					else
						tmp += i_lppInputBuffer[batchNum][inputNum] * this->layerData.lppNeuron[neuronNum][inputNum];
				}
				tmp += this->layerData.lpBias[neuronNum];

				if(!this->onUseDropOut && this->layerData.layerStructure.DropOut > 0.0f)
					tmp *= (1.0f - this->layerData.layerStructure.DropOut);

				// ������
				this->lpOutputBuffer[batchNum][neuronNum] = this->func_activation(tmp);

#ifdef _DEBUG
				if(isnan(this->lpOutputBuffer[batchNum][neuronNum]))
					return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
#endif
			}
		}

		// ��������̃f�[�^�����v�l�Ŋ���
		switch(this->layerData.layerStructure.ActivationType)
		{
		case Gravisbell::Layer::NeuralNetwork::FullyConnect_Activation::LayerStructure::ActivationType_softmax:
		case Gravisbell::Layer::NeuralNetwork::FullyConnect_Activation::LayerStructure::ActivationType_softmax_crossEntropy:
			{
				for(unsigned int batchNum=0; batchNum<this->batchSize; batchNum++)
				{
					// ���ς��Z�o
					F32 sum = 0.0f;
					for(unsigned int neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
					{
						sum += this->lpOutputBuffer[batchNum][neuronNum];
					}

					// �l�𕽋ϒl�Ŋ���
					for(unsigned int neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
					{
						if(sum == 0.0f)
							this->lpOutputBuffer[batchNum][neuronNum] = 1.0f / neuronNum;
						else
							this->lpOutputBuffer[batchNum][neuronNum] /= sum;
					
#ifdef _DEBUG
						if(isnan(this->lpOutputBuffer[batchNum][neuronNum]))
							return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
#endif
					}
				}
			}
			break;
		}


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �o�̓f�[�^�o�b�t�@���擾����.
		�z��̗v�f����GetOutputBufferCount�̖߂�l.
		@return �o�̓f�[�^�z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER FullyConnect_Activation_CPU::GetOutputBuffer()const
	{
		return &this->lppBatchOutputBuffer[0];
	}
	/** �o�̓f�[�^�o�b�t�@���擾����.
		@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
		@return ���������ꍇ0 */
	ErrorCode FullyConnect_Activation_CPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
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


	//================================
	// �w�K����
	//================================
	/** �w�K�덷���v�Z����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode FullyConnect_Activation_CPU::CalculateLearnError(CONST_BATCH_BUFFER_POINTER i_lppDOutputBufferPrev)
	{
		this->m_lppDOutputBufferPrev = i_lppDOutputBufferPrev;

		// �o�͌덷���v�Z
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			for(unsigned int neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
			{
				this->lpDOutputBuffer[batchNum][neuronNum] = this->func_dactivation(this->lpOutputBuffer[batchNum][neuronNum]) * i_lppDOutputBufferPrev[batchNum][neuronNum];

#ifdef _DEBUG
				if(isnan(this->lpDOutputBuffer[batchNum][neuronNum]))
					return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
#endif
			}
		}


#if 0
		for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
		{
			// �o�C�A�X�X�V
			{
				F32 sumDOutput = 0.0f;
				for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
				{
					 sumDOutput += this->lpDOutputBuffer[batchNum][neuronNum];
				}

#ifdef _DEBUG
				if(isnan(sumDOutput))
					return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
#endif

				this->layerData.lpBias[neuronNum] += this->learnData.LearnCoeff * sumDOutput;
			}

			// ���͑Ή��j���[�����X�V
			for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
			{
				if(this->onUseDropOut && this->lppDropOutBuffer[neuronNum][inputNum] == 0.0f)
					continue;

				F32 sumDOutput = 0.0f;
				for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
				{
					sumDOutput += this->m_lppInputBuffer[batchNum][inputNum] * this->lpDOutputBuffer[batchNum][neuronNum];
				}

#ifdef _DEBUG
				if(isnan(sumDOutput))
					return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
#endif

				this->layerData.lppNeuron[neuronNum][inputNum] += this->learnData.LearnCoeff * sumDOutput;
			}
		}
#endif



		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			// ���͌덷�������v�Z
			for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
			{
				float tmp = 0.0f;

				for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
				{
					if(this->onUseDropOut)
						tmp += this->lpDOutputBuffer[batchNum][neuronNum] * this->layerData.lppNeuron[neuronNum][inputNum] * this->lppDropOutBuffer[neuronNum][inputNum];
					else
						tmp += this->lpDOutputBuffer[batchNum][neuronNum] * this->layerData.lppNeuron[neuronNum][inputNum];
				}

				this->lpDInputBuffer[batchNum][inputNum] = tmp;

#ifdef _DEBUG
				if(isnan(this->lpDInputBuffer[batchNum][inputNum]))
					return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
#endif
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �w�K���������C���[�ɔ��f������.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		�o�͌덷�����A���͌덷�����͒��O��CalculateLearnError�̒l���Q�Ƃ���. */
	ErrorCode FullyConnect_Activation_CPU::ReflectionLearnError(void)
	{
#if 1
		for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
		{
			// �o�C�A�X�X�V
			{
				F32 sumDOutput = 0.0f;
				for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
				{
					 sumDOutput += this->lpDOutputBuffer[batchNum][neuronNum];
				}

#ifdef _DEBUG
				if(isnan(sumDOutput))
					return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
#endif

				this->layerData.lpBias[neuronNum] += this->learnData.LearnCoeff * sumDOutput;
			}

			// ���͑Ή��j���[�����X�V
			for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
			{
				if(this->onUseDropOut && this->lppDropOutBuffer[neuronNum][inputNum] == 0.0f)
					continue;

				F32 sumDOutput = 0.0f;
				for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
				{
					sumDOutput += this->m_lppInputBuffer[batchNum][inputNum] * this->lpDOutputBuffer[batchNum][neuronNum];
				}

#ifdef _DEBUG
				if(isnan(sumDOutput))
					return ErrorCode::ERROR_CODE_COMMON_CALCULATE_NAN;
#endif

				this->layerData.lppNeuron[neuronNum][inputNum] += this->learnData.LearnCoeff * sumDOutput;
			}
		}
#endif

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K�������擾����.
		�z��̗v�f����[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]
		@return	�덷�����z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER FullyConnect_Activation_CPU::GetDInputBuffer()const
	{
		return &this->lppBatchDInputBuffer[0];
	}
	/** �w�K�������擾����.
		@param lpDInputBuffer	�w�K�������i�[����z��.[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̔z�񂪕K�v */
	ErrorCode FullyConnect_Activation_CPU::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
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


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
