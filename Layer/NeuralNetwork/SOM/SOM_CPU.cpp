//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A������
// CPU�����p
//======================================
#include"stdafx.h"

#include"SOM_DATA.hpp"
#include"SOM_FUNC.hpp"
#include"SOM_Base.h"

#include"SOM_CPU.h"
#include"SOM_LayerData_CPU.h"


using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {



	/** �R���X�g���N�^ */
	SOM_CPU::SOM_CPU(Gravisbell::GUID guid, SOM_LayerData_CPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	SOM_Base				(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData				(i_layerData)	/**< ���C���[�f�[�^ */
		,	inputBufferCount		(0)		/**< ���̓o�b�t�@�� */
		,	unitCount				(0)		/**< �j���[������ */
		,	outputBufferCount		(0)		/**< �o�̓o�b�t�@�� */
		,	temporaryMemoryManager	(i_temporaryMemoryManager)
	{
	}
	/** �f�X�g���N�^ */
	SOM_CPU::~SOM_CPU()
	{
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 SOM_CPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode SOM_CPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	SOM_LayerData_Base& SOM_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const SOM_LayerData_Base& SOM_CPU::GetLayerData()const
	{
		return this->layerData;
	}


	//================================
	// ���Z����
	//================================
	/** ���Z�O���������s����.(�w�K�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
	ErrorCode SOM_CPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// ���͌덷/�o�͌덷�o�b�t�@�󂯎��p�̃A�h���X�z����쐬����

		// �p�����[�^�̕ω��ʃo�b�t�@
		this->lpDUnit.resize(this->unitCount * this->inputBufferCount);
		this->lppDUnit.resize(this->unitCount);
		for(U32 i=0; i<this->lppDUnit.size(); i++)
			this->lppDUnit[i] = (F32*)&this->lpDUnit[i * this->inputBufferCount];


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode SOM_CPU::PreProcessCalculate()
	{
		// ���̓o�b�t�@�����m�F
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;
		
		// �j���[���������m�F
		this->unitCount = this->GetUnitCount();
		if(this->unitCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;

		// �o�̓o�b�t�@�����m�F
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// �j���[�����o�b�t�@�̃T�C�Y�m�F
		if(this->layerData.lpUnitData.size() != this->unitCount * this->inputBufferCount)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;
		if(this->layerData.lppUnitData.size() != this->unitCount)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;

		// ����/�o�̓o�b�t�@�ۑ��p�̃A�h���X�z����쐬
		this->m_lppInputBuffer.resize(this->GetBatchSize(), NULL);
		this->m_lppOutputBuffer.resize(this->GetBatchSize(), NULL);

		// �e���j�b�g�̍��W���v�Z����
		this->lpUnitPos.resize(this->unitCount);
		for(U32 unitNo=0; unitNo<this->unitCount; unitNo++)
		{
			this->lpUnitPos[unitNo].resize(this->layerData.layerStructure.DimensionCount);

			U32 tmpNo = unitNo;
			for(U32 dimNo=0; dimNo<(U32)this->layerData.layerStructure.DimensionCount; dimNo++)
			{
				U32 pos = tmpNo % this->layerData.layerStructure.ResolutionCount;
				tmpNo /= this->layerData.layerStructure.ResolutionCount;

				this->lpUnitPos[unitNo][dimNo] = (F32)pos / (this->layerData.layerStructure.ResolutionCount - 1);
			}
		}

		// �ꎞ�o�b�t�@���m��


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���[�v�̏���������.�f�[�^�Z�b�g�̎��s�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode SOM_CPU::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}



	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode SOM_CPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// ����/�o�̓o�b�t�@�̃A�h���X��z��Ɋi�[
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->m_lppInputBuffer[batchNum]  = &i_lppInputBuffer[batchNum * this->inputBufferCount];
			this->m_lppOutputBuffer[batchNum] = &o_lppOutputBuffer[batchNum * this->outputBufferCount];
		}

		// BMU(Best Matching Unit)�𒲂ׂ�
		{
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				S32 bmuNo = -1;
				F32 bmuMatchRate = 0.0f;

				for(U32 unitNo=0; unitNo<this->unitCount; unitNo++)
				{
					// ���͐M���ƃ��j�b�g�̓��ς���v���Ƃ���(�{���ɂ����Ă�H)
					F32 matchRate=0;
					for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
					{
						matchRate += this->m_lppInputBuffer[batchNum][inputNum] * this->layerData.lppUnitData[unitNo][inputNum];
					}

					if(bmuNo<0 || matchRate>bmuMatchRate)
					{
						bmuNo = unitNo;
						bmuMatchRate = matchRate;
					}
				}

				// BMU�ԍ�����N�������W�ɕϊ�
				for(U32 dimNo=0; dimNo<(U32)this->layerData.layerStructure.DimensionCount; dimNo++)
					this->m_lppOutputBuffer[batchNum][dimNo] = this->lpUnitPos[bmuNo][dimNo];
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	//================================
	// �w�K����
	//================================
	/** ���͌덷�v�Z�������s����.�w�K�����ɓ��͌덷���擾�������ꍇ�Ɏg�p����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	o_lppDInputBuffer	���͌덷�����i�[�惌�C���[.	[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode SOM_CPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		if(o_lppDInputBuffer)
		{
			// ���͌덷�o�b�t�@���N���A
			memset(o_lppDInputBuffer, 0, sizeof(F32)*this->inputBufferCount*this->GetBatchSize());
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode SOM_CPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// ����/�o�̓o�b�t�@�̃A�h���X��z��Ɋi�[
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->m_lppInputBuffer[batchNum]   = &i_lppInputBuffer[batchNum * this->inputBufferCount];
			this->m_lppOutputBuffer[batchNum]  = const_cast<BATCH_BUFFER_POINTER>(&i_lppOutputBuffer[batchNum * this->outputBufferCount]);
		}

		// �덷���v�Z
		{
			// ���Ԍ��������v�Z
			F32 timeAttenuationRate = this->GetRuntimeParameterByStructure().SOM_L0 * exp(-(S32)this->layerData.learnTime / this->GetRuntimeParameterByStructure().SOM_ramda);
			// �����������v�Z�ʂ̌W�����v�Z
			F32 lengthAttenuationRateCoeff = 2.0f * pow(this->GetRuntimeParameterByStructure().SOM_sigma * exp(-(S32)this->layerData.learnTime / this->GetRuntimeParameterByStructure().SOM_ramda), 2);

			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				// ���j�b�g���ƂɌ덷���v�Z
				for(U32 unitNo=0; unitNo<this->layerData.lppUnitData.size(); unitNo++)
				{
					// BMU�Ƃ̋�����2������߂�
					F32 length2 = 0.0f;
					for(U32 dimNo=0; dimNo<(U32)this->layerData.layerStructure.DimensionCount; dimNo++)
					{
						length2 += pow(this->m_lppOutputBuffer[batchNum][dimNo] - this->lpUnitPos[unitNo][dimNo], 2);
					}

					// ���������������߂�
					F32 lengthAttenuationRate = exp(-length2 / lengthAttenuationRateCoeff);

					// �����������߂�
					F32 attenuationRate = timeAttenuationRate * lengthAttenuationRate;

					// �덷�̍X�V
					for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
					{
						F32 dValue = this->m_lppInputBuffer[batchNum][inputNum] - this->layerData.lppUnitData[unitNo][inputNum];

						this->lppDUnit[unitNo][inputNum] += attenuationRate * dValue;
					}
				}
			}
		}

		// �덷�����j�b�g�ɉ��Z
		for(int i=0; i<this->layerData.lpUnitData.size(); i++)
		{
			this->layerData.lpUnitData[i] += this->lpDUnit[i] / this->GetBatchSize();
		}

		// �w�K�񐔂��J�E���g�A�b�v
		this->layerData.learnTime++;

		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
