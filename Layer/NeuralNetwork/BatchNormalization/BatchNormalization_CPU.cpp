//======================================
// �o�b�`���K�����C���[
// CPU�����p
//======================================
#include"stdafx.h"

#include"BatchNormalization_DATA.hpp"
#include"BatchNormalization_FUNC.hpp"
#include"BatchNormalization_Base.h"

#include"BatchNormalization_CPU.h"
#include"BatchNormalization_LayerData_CPU.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** �R���X�g���N�^ */
	BatchNormalization_CPU::BatchNormalization_CPU(Gravisbell::GUID guid, BatchNormalization_LayerData_CPU& i_layerData)
		:	BatchNormalization_Base	(guid)
		,	layerData				(i_layerData)	/**< ���C���[�f�[�^ */
		,	inputBufferCount		(0)				/**< ���̓o�b�t�@�� */
		,	outputBufferCount		(0)				/**< �o�̓o�b�t�@�� */
		,	channeclBufferCount		(0)				/**< 1�`�����l��������̃o�b�t�@�� */
		,	onLearnMode				(false)			/**< �w�K�������t���O */
		,	learnCount				(0)				/**< �w�K���s�� */
	{
	}
	/** �f�X�g���N�^ */
	BatchNormalization_CPU::~BatchNormalization_CPU()
	{
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 BatchNormalization_CPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode BatchNormalization_CPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	BatchNormalization_LayerData_Base& BatchNormalization_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const BatchNormalization_LayerData_Base& BatchNormalization_CPU::GetLayerData()const
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
	ErrorCode BatchNormalization_CPU::PreProcessLearn(unsigned int batchSize)
	{
		ErrorCode errorCode = this->PreProcessCalculate(batchSize);
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// �w�K�p�̕ϐ����쐬
		this->onLearnMode = true;
		this->learnCount = 0;
		this->lpTmpMean.resize(this->layerData.inputDataStruct.ch, 0.0f);
		this->lpTmpVariance.resize(this->layerData.inputDataStruct.ch, 0.0f);

		// �o�͌덷�o�b�t�@�󂯎��p�̃A�h���X�z����쐬����
		this->m_lppDOutputBufferPrev.resize(batchSize);

		// ���͌덷�o�b�t�@�󂯎��p�̃A�h���X�z����쐬����
		this->m_lppDInputBuffer.resize(batchSize);


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode BatchNormalization_CPU::PreProcessCalculate(unsigned int batchSize)
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

		// �`�����l�����Ƃ̃o�b�t�@�����m�F
		this->channeclBufferCount = this->layerData.inputDataStruct.z * this->layerData.inputDataStruct.y * this->layerData.inputDataStruct.x;
		if(this->channeclBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// ���̓o�b�t�@�ۑ��p�̃A�h���X�z����쐬
		this->m_lppInputBuffer.resize(batchSize, NULL);

		// �o�̓o�b�t�@���쐬
		this->lpOutputBuffer.resize(this->batchSize * this->outputBufferCount);
		this->lppBatchOutputBuffer.resize(this->batchSize);
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			this->lppBatchOutputBuffer[batchNum] = &this->lpOutputBuffer[batchNum * this->outputBufferCount];
		}

		// ����,���U���ꎞ�o�b�t�@�Ɉڂ�
		this->lpTmpMean = this->layerData.lpMean;
		this->lpTmpVariance = this->layerData.lpVariance;

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode BatchNormalization_CPU::PreProcessLearnLoop(const SettingData::Standard::IData& data)
	{
		// �w�K�ݒ��ۑ�
		if(this->pLearnData != NULL)
			delete this->pLearnData;
		this->pLearnData = data.Clone();

		this->pLearnData->WriteToStruct((BYTE*)&learnData);


		// �w�K�񐔂�������
		this->learnCount = 0;

		// ���Z�p�̕���.���U��������
		for(U32 ch=0; ch<this->layerData.inputDataStruct.ch; ch++)
		{
			this->layerData.lpMean[ch] = 0.0f;
			this->layerData.lpVariance[ch] = 0.0f;
		}

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}
	/** ���Z���[�v�̏���������.�f�[�^�Z�b�g�̉��Z�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode BatchNormalization_CPU::PreProcessCalculateLoop()
	{
		// ����,���U���ꎞ�o�b�t�@�Ɉڂ�
		this->lpTmpMean = this->layerData.lpMean;
		this->lpTmpVariance = this->layerData.lpVariance;

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode BatchNormalization_CPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
	{
		// ���̓o�b�t�@�̃A�h���X��z��Ɋi�[
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			this->m_lppInputBuffer[batchNum] = &i_lpInputBuffer[batchNum * this->inputBufferCount];

		// �w�K���Ȃ�Ε��ρA���U�����߂�
		if(this->onLearnMode)
		{
			// ���ς����߂�
			for(U32 ch=0; ch<this->layerData.inputDataStruct.ch; ch++)
			{
				this->lpTmpMean[ch] = 0.0f;
				for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
				{
					for(U32 bufNum=0; bufNum<this->channeclBufferCount; bufNum++)
					{
						this->lpTmpMean[ch] += this->m_lppInputBuffer[batchNum][this->channeclBufferCount*ch + bufNum];
					}
				}
				this->lpTmpMean[ch] /= (this->batchSize * this->channeclBufferCount);
			}

			// ���U�����߂�
			for(U32 ch=0; ch<this->layerData.inputDataStruct.ch; ch++)
			{
				F64 tmp = 0.0f;
				for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
				{
					for(U32 bufNum=0; bufNum<this->channeclBufferCount; bufNum++)
					{
						F64 value = this->m_lppInputBuffer[batchNum][this->channeclBufferCount*ch + bufNum];

						tmp += (value - this->lpTmpMean[ch]) * (value - this->lpTmpMean[ch]);
					}
				}
				this->lpTmpVariance[ch]    = (F32)(tmp / (this->batchSize * this->channeclBufferCount));
			}
		}

		// ����,���U�𗘗p���Đ��K��
		for(U32 ch=0; ch<this->layerData.inputDataStruct.ch; ch++)
		{
			F32 mean = this->lpTmpMean[ch];
			F32 variance = this->lpTmpVariance[ch] + (F32)max(this->layerData.layerStructure.epsilon, 1e-5);
			F32 sqrtVariance = (F32)sqrt(variance);

			for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			{
				for(U32 bufNum=0; bufNum<this->channeclBufferCount; bufNum++)
				{
					F32 value = this->m_lppInputBuffer[batchNum][this->channeclBufferCount*ch + bufNum];

					// ���K��
					F32 value2 = (value - mean) / sqrtVariance;

					// �X�P�[�����O�ƃo�C�A�X
					this->lppBatchOutputBuffer[batchNum][this->channeclBufferCount*ch + bufNum] = this->layerData.lpScale[ch] * value2 + this->layerData.lpBias[ch];
				}
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �o�̓f�[�^�o�b�t�@���擾����.
		�z��̗v�f����GetOutputBufferCount�̖߂�l.
		@return �o�̓f�[�^�z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER BatchNormalization_CPU::GetOutputBuffer()const
	{
		return &this->lpOutputBuffer[0];
	}
	/** �o�̓f�[�^�o�b�t�@���擾����.
		@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
		@return ���������ꍇ0 */
	ErrorCode BatchNormalization_CPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
	{
		if(o_lpOutputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 outputBufferCount = this->GetOutputBufferCount();

		CONST_BATCH_BUFFER_POINTER lppUseOutputBuffer = this->GetOutputBuffer();

		memcpy(o_lpOutputBuffer, this->GetOutputBuffer(), sizeof(F32) * outputBufferCount * batchSize);

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
	ErrorCode BatchNormalization_CPU::CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// �o�͌덷�o�b�t�@�̃A�h���X��z��Ɋi�[
		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			this->m_lppDOutputBufferPrev[batchNum] = &i_lppDOutputBuffer[batchNum * this->outputBufferCount];

		// ���͌덷�o�b�t�@�̃A�h���X��z��Ɋi�[
		this->m_lpDInputBuffer = o_lppDInputBuffer;
		if(o_lppDInputBuffer)
		{
			for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
				this->m_lppDInputBuffer[batchNum] = &o_lppDInputBuffer[batchNum * this->inputBufferCount];
		}


		for(U32 ch=0; ch<this->layerData.inputDataStruct.ch; ch++)
		{
			F32 mean = this->lpTmpMean[ch];
			F32 variance = this->lpTmpVariance[ch] + (F32)max(this->layerData.layerStructure.epsilon, 1e-5);
			F32 sqrtVariance  = (F32)sqrt(variance);
			F32 sqrtVariance3 = sqrtVariance*sqrtVariance*sqrtVariance;
			F32 scale = this->layerData.lpScale[ch];

			// ���ςƕ��U�̌덷�̍��v�l�����߂�
			F32 dMean = 0.0f;
			F32 dVariance = 0.0f;
			for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			{
				for(U32 bufNum=0; bufNum<this->channeclBufferCount; bufNum++)
				{
					F32 value  = this->m_lppInputBuffer[batchNum][this->channeclBufferCount*ch + bufNum];
					F32 dOutput = this->m_lppDOutputBufferPrev[batchNum][this->channeclBufferCount*ch + bufNum];
					F32 dValue2 = dOutput * scale;

					dVariance += dValue2 * (value - mean) * (-1) / 2 / sqrtVariance3;
					dMean     += dValue2 * (-1) / sqrtVariance;
				}
			}

			// ���͌덷�����߂�
			if(o_lppDInputBuffer)
			{
				for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
				{
					for(U32 bufNum=0; bufNum<this->channeclBufferCount; bufNum++)
					{
						F32 value  = this->m_lppInputBuffer[batchNum][this->channeclBufferCount*ch + bufNum];
						F32 dOutput = this->m_lppDOutputBufferPrev[batchNum][this->channeclBufferCount*ch + bufNum];
						F32 dValue2 = dOutput * scale;

						this->m_lppDInputBuffer[batchNum][this->channeclBufferCount*ch + bufNum]
							= dValue2 / sqrtVariance
							+ dVariance * 2 * (value - mean) / (this->channeclBufferCount * this->batchSize)
							+ dMean / (this->channeclBufferCount * this->batchSize);
					}
				}
			}
		}


		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode BatchNormalization_CPU::Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		Gravisbell::ErrorCode errCode = this->CalculateDInput(o_lppDInputBuffer, i_lppDOutputBuffer);
		if(errCode != ErrorCode::ERROR_CODE_NONE)
			return errCode;


		// ���ϒl�X�V�p�̌W�����Z�o
		F64 factor = 1.0 / (this->learnCount+1);

		// �w�K�����̎��s�񐔂��J�E���g�A�b�v
		this->learnCount++;

		for(U32 ch=0; ch<this->layerData.inputDataStruct.ch; ch++)
		{
			F32 mean = this->lpTmpMean[ch];
			F32 variance = this->lpTmpVariance[ch] + (F32)max(this->layerData.layerStructure.epsilon, 1e-5);
			F64 sqrtVariance = (F32)sqrt(variance);
			F64 sqrtVarianceInv = 1.0f / sqrtVariance;

			F32 dBias = 0.0f;
			F32 dScale = 0.0f;

			for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			{
				for(U32 bufNum=0; bufNum<this->channeclBufferCount; bufNum++)
				{
					F32 value = this->m_lppInputBuffer[batchNum][this->channeclBufferCount*ch + bufNum];

					// ���K��
					F32 value2 = (F32)( (value - mean) * sqrtVarianceInv );

					dScale += this->m_lppDOutputBufferPrev[batchNum][this->channeclBufferCount*ch + bufNum] * value2;
					dBias  += this->m_lppDOutputBufferPrev[batchNum][this->channeclBufferCount*ch + bufNum];
				}
			}

			// �l���X�V
			this->layerData.lpScale[ch] += this->learnData.LearnCoeff * dScale;
			this->layerData.lpBias[ch]  += this->learnData.LearnCoeff * dBias;

			// ���ςƕ��U���X�V
			this->layerData.lpMean[ch]     = (F32)((1.0 - factor) * this->layerData.lpMean[ch]     + factor * this->lpTmpMean[ch]);
			this->layerData.lpVariance[ch] = (F32)((1.0 - factor) * this->layerData.lpVariance[ch] + factor * variance);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �w�K�������擾����.
		�z��̗v�f����[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]
		@return	�덷�����z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER BatchNormalization_CPU::GetDInputBuffer()const
	{
		return this->m_lpDInputBuffer;
	}
	/** �w�K�������擾����.
		@param lpDInputBuffer	�w�K�������i�[����z��.[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̔z�񂪕K�v */
	ErrorCode BatchNormalization_CPU::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
	{
		if(o_lpDInputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 inputBufferCount = this->GetInputBufferCount();

		CONST_BATCH_BUFFER_POINTER lppUseDInputBuffer = this->GetDInputBuffer();

		memcpy(o_lpDInputBuffer, this->GetDInputBuffer(), sizeof(F32) * inputBufferCount * batchSize);

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
