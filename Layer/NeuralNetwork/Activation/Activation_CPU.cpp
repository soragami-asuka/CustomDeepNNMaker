//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A������
// CPU�����p
//======================================
#include"stdafx.h"

#include"Activation_DATA.hpp"
#include"Activation_FUNC.hpp"
#include"Activation_Base.h"

#include"Activation_CPU.h"
#include"Activation_LayerData_CPU.h"

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

	// tanh�n
	F32 func_activation_tanh(F32 x)
	{
		return tanh(x);
	}
	F32 func_dactivation_tanh(F32 x)
	{
		return 1.0f - x*x;
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
	Activation_CPU::Activation_CPU(Gravisbell::GUID guid, Activation_LayerData_CPU& i_layerData)
		:	Activation_Base	(guid)
		,	layerData						(i_layerData)	/**< ���C���[�f�[�^ */
		,	inputBufferCount				(0)		/**< ���̓o�b�t�@�� */
		,	outputBufferCount				(0)		/**< �o�̓o�b�t�@�� */
		,	func_activation					(func_activation_sigmoid)
		,	func_dactivation				(func_dactivation_sigmoid)
	{
	}
	/** �f�X�g���N�^ */
	Activation_CPU::~Activation_CPU()
	{
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 Activation_CPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode Activation_CPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	Activation_LayerData_Base& Activation_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const Activation_LayerData_Base& Activation_CPU::GetLayerData()const
	{
		return this->layerData;
	}


	//===========================
	// ���C���[�ۑ�
	//===========================
	/** ���C���[���o�b�t�@�ɏ�������.
		@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
		@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
	S32 Activation_CPU::WriteToBuffer(BYTE* o_lpBuffer)const
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
	ErrorCode Activation_CPU::PreProcessLearn(unsigned int batchSize)
	{
		ErrorCode errorCode = this->PreProcessCalculate(batchSize);
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// �o�͌덷�o�b�t�@�󂯎��p�̃A�h���X�z����쐬����
		this->m_lppDOutputBufferPrev.resize(batchSize);

		// ���͍����o�b�t�@���쐬
		switch(this->layerData.layerStructure.ActivationType)
		{
			// lenear
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_lenear:
			break;

		default:
			this->lpDInputBuffer.resize(this->batchSize * this->inputBufferCount);
			this->lppBatchDInputBuffer.resize(this->batchSize);
			for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			{
				this->lppBatchDInputBuffer[batchNum] = &this->lpDInputBuffer[batchNum * this->inputBufferCount];
			}
			break;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode Activation_CPU::PreProcessCalculate(unsigned int batchSize)
	{
		this->batchSize = batchSize;

		// ���̓o�b�t�@�����m�F
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// �o�̓o�b�t�@�����m�F
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// ���̓o�b�t�@�ۑ��p�̃A�h���X�z����쐬
		this->m_lppInputBuffer.resize(batchSize, NULL);

		// �o�̓o�b�t�@���쐬
		switch(this->layerData.layerStructure.ActivationType)
		{
			// lenear
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_lenear:
			break;

		default:
			this->lpOutputBuffer.resize(this->batchSize * this->outputBufferCount);
			this->lppBatchOutputBuffer.resize(this->batchSize);
			for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			{
				this->lppBatchOutputBuffer[batchNum] = &this->lpOutputBuffer[batchNum * this->outputBufferCount];
			}
			break;
		}


		// �������֐���ݒ�
		switch(this->layerData.layerStructure.ActivationType)
		{
			// lenear
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_lenear:
			this->func_activation  = ::func_activation_lenear;
			this->func_dactivation = ::func_dactivation_lenear;
			break;

			// Sigmoid
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_sigmoid:
		default:
			this->func_activation  = ::func_activation_sigmoid;
			this->func_dactivation = ::func_dactivation_sigmoid;
			break;
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_sigmoid_crossEntropy:
			this->func_activation  = ::func_activation_sigmoid_crossEntropy;
			this->func_dactivation = ::func_dactivation_sigmoid_crossEntropy;
			break;

			// ReLU
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_ReLU:
			this->func_activation  = ::func_activation_ReLU;
			this->func_dactivation = ::func_dactivation_ReLU;
			break;

			// SoftMax�n
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_ALL:
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_CH:
			this->func_activation  = ::func_activation_SoftMax;
			this->func_dactivation = ::func_dactivation_sigmoid;
			break;
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_ALL_crossEntropy:
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_CH_crossEntropy:
			this->func_activation  = ::func_activation_SoftMax;
			this->func_dactivation = ::func_dactivation_sigmoid_crossEntropy;
			break;
		}

		// ���Z�p�̃o�b�t�@���m��
		switch(this->layerData.layerStructure.ActivationType)
		{
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_CH:
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_CH_crossEntropy:
			this->lpCalculateSum.resize(this->layerData.inputDataStruct.z * this->layerData.inputDataStruct.y * this->layerData.inputDataStruct.x);
			break;
		default:
			this->lpCalculateSum.clear();
			break;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode Activation_CPU::PreProcessLearnLoop(const SettingData::Standard::IData& data)
	{
		if(this->pLearnData != NULL)
			delete this->pLearnData;
		this->pLearnData = data.Clone();

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}
	/** ���Z���[�v�̏���������.�f�[�^�Z�b�g�̉��Z�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode Activation_CPU::PreProcessCalculateLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode Activation_CPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
	{
		// ���̓o�b�t�@�̃A�h���X��z��Ɋi�[
		this->m_lpInputBuffer = i_lpInputBuffer;
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			this->m_lppInputBuffer[batchNum] = &i_lpInputBuffer[batchNum * this->inputBufferCount];

		
		switch(this->layerData.layerStructure.ActivationType)
		{
			// lenear
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_lenear:
			break;

		default:
			for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			{
				for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
				{
					// ������
					this->lppBatchOutputBuffer[batchNum][inputNum] = this->func_activation(this->m_lppInputBuffer[batchNum][inputNum]);
				}
			}
			break;
		}

		// softMax�����s����
		switch(this->layerData.layerStructure.ActivationType)
		{
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_ALL:
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_ALL_crossEntropy:
			{
				for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
				{
					// ���v�l���Z�o
					F32 sum = 0.0f;
					for(U32 outputNum=0; outputNum<this->outputBufferCount; outputNum++)
					{
						sum += this->lppBatchOutputBuffer[batchNum][outputNum];
					}

					// �l�����v�l�Ŋ���
					for(U32 outputNum=0; outputNum<this->outputBufferCount; outputNum++)
					{
						if(sum == 0.0f)
							this->lppBatchOutputBuffer[batchNum][outputNum] = 1.0f / this->outputBufferCount;
						else
							this->lppBatchOutputBuffer[batchNum][outputNum] /= sum;
					}
				}
			}
			break;
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_CH:
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_softmax_CH_crossEntropy:
			{
				U32 chSize = this->layerData.inputDataStruct.z * this->layerData.inputDataStruct.y * this->layerData.inputDataStruct.x;

				for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
				{
					// �ꎞ�o�b�t�@�N���A
					memset(&this->lpCalculateSum[0], 0, this->lpCalculateSum.size()*sizeof(F32));

					// ���v�l���Z�o
					for(U32 ch=0; ch<this->layerData.inputDataStruct.ch; ch++)
					{
						for(U32 z=0; z<this->layerData.inputDataStruct.z; z++)
						{
							for(U32 y=0; y<this->layerData.inputDataStruct.y; y++)
							{
								for(U32 x=0; x<this->layerData.inputDataStruct.x; x++)
								{
									U32 offset = (((((ch*this->layerData.inputDataStruct.z+z)*this->layerData.inputDataStruct.y)+y)*this->layerData.inputDataStruct.x)+x);

									this->lpCalculateSum[offset] += this->lppBatchOutputBuffer[batchNum][ch*chSize + offset];
								}
							}
						}
					}

					// ���v�l�Ŋ���
					for(U32 ch=0; ch<this->layerData.inputDataStruct.ch; ch++)
					{
						for(U32 z=0; z<this->layerData.inputDataStruct.z; z++)
						{
							for(U32 y=0; y<this->layerData.inputDataStruct.y; y++)
							{
								for(U32 x=0; x<this->layerData.inputDataStruct.x; x++)
								{
									U32 offset = (((((ch*this->layerData.inputDataStruct.z+z)*this->layerData.inputDataStruct.y)+y)*this->layerData.inputDataStruct.x)+x);

									this->lppBatchOutputBuffer[batchNum][ch*chSize + offset] /= this->lpCalculateSum[offset];
								}
							}
						}
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
	CONST_BATCH_BUFFER_POINTER Activation_CPU::GetOutputBuffer()const
	{
		switch(this->layerData.layerStructure.ActivationType)
		{
			// lenear
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_lenear:
			return this->m_lpInputBuffer;
			break;

		default:
			return &this->lpOutputBuffer[0];
		}
	}
	/** �o�̓f�[�^�o�b�t�@���擾����.
		@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
		@return ���������ꍇ0 */
	ErrorCode Activation_CPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
	{
		if(o_lpOutputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 outputBufferCount = this->GetOutputBufferCount();

		memcpy(o_lpOutputBuffer, this->GetOutputBuffer(), sizeof(F32)*outputBufferCount*batchSize);

		return ErrorCode::ERROR_CODE_NONE;
	}


	//================================
	// �w�K����
	//================================
	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode Activation_CPU::Training(CONST_BATCH_BUFFER_POINTER i_lpDOutputBufferPrev)
	{
		// �o�͌덷�o�b�t�@�̃A�h���X��z��Ɋi�[
		this->m_lpDOutputBufferPrev = i_lpDOutputBufferPrev;
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			this->m_lppDOutputBufferPrev[batchNum] = &i_lpDOutputBufferPrev[batchNum * this->outputBufferCount];


		switch(this->layerData.layerStructure.ActivationType)
		{
			// lenear
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_lenear:
			break;

		default:
			// �o�͌덷���v�Z
			for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			{
				for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
				{
					this->lppBatchDInputBuffer[batchNum][inputNum] = this->func_dactivation(this->lppBatchOutputBuffer[batchNum][inputNum]) * this->m_lppDOutputBufferPrev[batchNum][inputNum];
				}
			}
			break;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �w�K�������擾����.
		�z��̗v�f����[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]
		@return	�덷�����z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER Activation_CPU::GetDInputBuffer()const
	{
		switch(this->layerData.layerStructure.ActivationType)
		{
			// lenear
		case Gravisbell::Layer::NeuralNetwork::Activation::LayerStructure::ActivationType_lenear:
			return this->m_lpDOutputBufferPrev;
			break;

		default:
			return &this->lpDInputBuffer[0];
			break;
		}
	}
	/** �w�K�������擾����.
		@param lpDInputBuffer	�w�K�������i�[����z��.[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̔z�񂪕K�v */
	ErrorCode Activation_CPU::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
	{
		if(o_lpDInputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 inputBufferCount = this->GetInputBufferCount();

		memcpy(o_lpDInputBuffer, this->GetDInputBuffer(), sizeof(F32)*inputBufferCount*batchSize);

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
