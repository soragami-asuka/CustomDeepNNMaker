//======================================
// �o�b�`���K�����C���[
// CPU�����p
//======================================
#include"stdafx.h"

#include"BatchNormalizationAll_DATA.hpp"
#include"BatchNormalizationAll_FUNC.hpp"
#include"BatchNormalizationAll_Base.h"

#include"BatchNormalizationAll_CPU.h"
#include"BatchNormalizationAll_LayerData_CPU.h"


using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** �R���X�g���N�^ */
	BatchNormalizationAll_CPU::BatchNormalizationAll_CPU(Gravisbell::GUID guid, BatchNormalizationAll_LayerData_CPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	BatchNormalizationAll_Base	(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData				(i_layerData)	/**< ���C���[�f�[�^ */
		,	inputBufferCount		(0)				/**< ���̓o�b�t�@�� */
		,	outputBufferCount		(0)				/**< �o�̓o�b�t�@�� */
		,	onLearnMode				(false)			/**< �w�K�������t���O */
		,	learnCount				(0)				/**< �w�K���s�� */
	{
	}
	/** �f�X�g���N�^ */
	BatchNormalizationAll_CPU::~BatchNormalizationAll_CPU()
	{
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 BatchNormalizationAll_CPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_CPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode BatchNormalizationAll_CPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	ILayerData& BatchNormalizationAll_CPU::GetLayerData()
	{
		return this->layerData;
	}
	const ILayerData& BatchNormalizationAll_CPU::GetLayerData()const
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
	ErrorCode BatchNormalizationAll_CPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// �w�K�p�̕ϐ����쐬
		this->onLearnMode = true;
		this->learnCount = 0;
		this->lpTmpMean.resize(this->layerData.lpMean.size(), 0.0f);
		this->lpTmpVariance.resize(this->layerData.lpVariance.size(), 0.0f);

		// ���͌덷/�o�͌덷�o�b�t�@�󂯎��p�̃A�h���X�z����쐬����
		this->lppBatchDOutputBuffer.resize(this->GetBatchSize());
		this->lppBatchDInputBuffer.resize(this->GetBatchSize());

		// �p�����[�^�ω��ʂ̃o�b�t�@���m��
		this->lpDBias.resize(this->layerData.lpBias.size());
		this->lpDScale.resize(this->layerData.lpScale.size());

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode BatchNormalizationAll_CPU::PreProcessCalculate()
	{
		// ���̓o�b�t�@�����m�F
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// �o�̓o�b�t�@�����m�F
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// ����/�o�̓o�b�t�@�ۑ��p�̃A�h���X�z����쐬
		this->lppBatchInputBuffer.resize(this->GetBatchSize(), NULL);
		this->lppBatchOutputBuffer.resize(this->GetBatchSize());

		// ����,���U���ꎞ�o�b�t�@�Ɉڂ�
		this->lpTmpMean = this->layerData.lpMean;
		this->lpTmpVariance = this->layerData.lpVariance;

		return ErrorCode::ERROR_CODE_NONE;
	}

	
	/** ���[�v�̏���������.�f�[�^�Z�b�g�̎��s�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode BatchNormalizationAll_CPU::PreProcessLoop()
	{
		switch(this->GetProcessType())
		{
		case ProcessType::PROCESSTYPE_LEARN:
			{
				// �w�K�񐔂�������
				this->learnCount = 0;

				// ���Z�p�̕���.���U��������
				for(U32 ch=0; ch<1; ch++)
				{
					this->layerData.lpMean[ch] = 0.0f;
					this->layerData.lpVariance[ch] = 0.0f;
				}
			}
			break;
		case ProcessType::PROCESSTYPE_CALCULATE:	
			{
				// ����,���U���ꎞ�o�b�t�@�Ɉڂ�
				this->lpTmpMean = this->layerData.lpMean;
				this->lpTmpVariance = this->layerData.lpVariance;
			}
			break;
		}

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}

	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode BatchNormalizationAll_CPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// ����/�o�̓o�b�t�@�̃A�h���X��z��Ɋi�[
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->lppBatchInputBuffer[batchNum]  = &i_lppInputBuffer[batchNum * this->inputBufferCount];
			this->lppBatchOutputBuffer[batchNum] = &o_lppOutputBuffer[batchNum * this->outputBufferCount];
		}

		// �w�K���Ȃ�Ε��ρA���U�����߂�
		if(this->onLearnMode)
		{
			// ���ς����߂�
			{
				this->lpTmpMean[0] = 0.0f;
				for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
				{
					for(U32 bufNum=0; bufNum<this->outputBufferCount; bufNum++)
					{
						this->lpTmpMean[0] += this->lppBatchInputBuffer[batchNum][bufNum];
					}
				}
				this->lpTmpMean[0] /= (this->GetBatchSize() * this->outputBufferCount);
			}

			// ���U�����߂�
			{
				F64 tmp = 0.0f;
				for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
				{
					for(U32 bufNum=0; bufNum<this->outputBufferCount; bufNum++)
					{
						F64 value = this->lppBatchInputBuffer[batchNum][bufNum];

						tmp += (value - this->lpTmpMean[0]) * (value - this->lpTmpMean[0]);
					}
				}
				this->lpTmpVariance[0] = (F32)(tmp / (this->GetBatchSize() * this->outputBufferCount));
			}
		}

		// ����,���U�𗘗p���Đ��K��
		{
			F32 mean = this->lpTmpMean[0];
			F32 variance = this->lpTmpVariance[0] + (F32)max(this->layerData.layerStructure.epsilon, 1e-5);
			F32 sqrtVariance = (F32)sqrt(variance);

			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				for(U32 bufNum=0; bufNum<this->outputBufferCount; bufNum++)
				{
					F32 value = this->lppBatchInputBuffer[batchNum][bufNum];

					// ���K��
					F32 value2 = (value - mean) / sqrtVariance;

					// �X�P�[�����O�ƃo�C�A�X
					this->lppBatchOutputBuffer[batchNum][bufNum] = this->layerData.lpScale[0] * value2 + this->layerData.lpBias[0];
				}
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
	ErrorCode BatchNormalizationAll_CPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// ����/�o�̓o�b�t�@�̃A�h���X��z��Ɋi�[
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->lppBatchInputBuffer[batchNum]  = &i_lppInputBuffer[batchNum * this->inputBufferCount];
			this->lppBatchOutputBuffer[batchNum] = const_cast<BATCH_BUFFER_POINTER>(&i_lppOutputBuffer[batchNum * this->outputBufferCount]);
		}

		// ���͌덷�o�b�t�@�̃A�h���X��z��Ɋi�[�z��Ɋi�[
		if(o_lppDInputBuffer)
		{
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				this->lppBatchDInputBuffer[batchNum]  = &o_lppDInputBuffer[batchNum * this->inputBufferCount];
			}
		}

		// �o�͌덷�o�b�t�@�̃A�h���X��z��Ɋi�[
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->lppBatchDOutputBuffer[batchNum] = &i_lppDOutputBuffer[batchNum * this->outputBufferCount];
		}


		{
			F32 mean = this->lpTmpMean[0];
			F32 variance = this->lpTmpVariance[0] + (F32)max(this->layerData.layerStructure.epsilon, 1e-5);
			F32 sqrtVariance  = (F32)sqrt(variance);
			F32 sqrtVariance3 = sqrtVariance*sqrtVariance*sqrtVariance;
			F32 scale = this->layerData.lpScale[0];

			// ���ςƕ��U�̌덷�̍��v�l�����߂�
			F32 dMean = 0.0f;
			F32 dVariance = 0.0f;
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				for(U32 bufNum=0; bufNum<this->outputBufferCount; bufNum++)
				{
					F32 value  = this->lppBatchInputBuffer[batchNum][bufNum];
					F32 dOutput = this->lppBatchDOutputBuffer[batchNum][bufNum];
					F32 dValue2 = dOutput * scale;

					dVariance += dValue2 * (value - mean) * (-1) / 2 / sqrtVariance3;
					dMean     += dValue2 * (-1) / sqrtVariance;
				}
			}

			// ���͌덷�����߂�
			if(o_lppDInputBuffer)
			{
				for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
				{
					for(U32 bufNum=0; bufNum<this->outputBufferCount; bufNum++)
					{
						F32 value  = this->lppBatchInputBuffer[batchNum][bufNum];
						F32 dOutput = this->lppBatchDOutputBuffer[batchNum][bufNum];
						F32 dValue2 = dOutput * scale;

						this->lppBatchDInputBuffer[batchNum][bufNum]
							= dValue2 / sqrtVariance
							+ dVariance * 2 * (value - mean) / (this->outputBufferCount * this->GetBatchSize())
							+ dMean / (this->outputBufferCount * this->GetBatchSize());
					}
				}
			}
		}


#ifdef _DEBUG
		std::vector<float> lpTmpInputBuffer(this->GetBatchSize() * this->inputBufferCount);
		memcpy(&lpTmpInputBuffer[0], i_lppInputBuffer, sizeof(float)*lpTmpInputBuffer.size());

		std::vector<float> lpTmpOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
		memcpy(&lpTmpOutputBuffer[0], i_lppOutputBuffer, sizeof(float)*lpTmpOutputBuffer.size());

		std::vector<float> lpTmpDOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
		memcpy(&lpTmpDOutputBuffer[0], i_lppDOutputBuffer, sizeof(float)*lpTmpDOutputBuffer.size());

		std::vector<float> lpTmpDInputBuffer(this->GetBatchSize() * this->inputBufferCount);
		memcpy(&lpTmpDInputBuffer[0], o_lppDInputBuffer, sizeof(float)*lpTmpDInputBuffer.size());
#endif


		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode BatchNormalizationAll_CPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		Gravisbell::ErrorCode errCode = this->CalculateDInput(o_lppDInputBuffer, i_lppDOutputBuffer);
		if(errCode != ErrorCode::ERROR_CODE_NONE)
			return errCode;


		// ���ϒl�X�V�p�̌W�����Z�o
		F64 factor = max(1.0 / (this->learnCount+1), this->GetRuntimeParameterByStructure().AverageUpdateCoeffMin);

		// �w�K�����̎��s�񐔂��J�E���g�A�b�v
		this->learnCount++;

		{
			F32 mean = this->lpTmpMean[0];
			F32 variance = this->lpTmpVariance[0] + (F32)max(this->layerData.layerStructure.epsilon, 1e-5);
			F64 sqrtVariance = (F32)sqrt(variance);
			F64 sqrtVarianceInv = 1.0f / sqrtVariance;

			this->lpDScale[0] = 0.0f;
			this->lpDBias[0]  = 0.0f;

			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				for(U32 bufNum=0; bufNum<this->outputBufferCount; bufNum++)
				{
					F32 value = this->lppBatchInputBuffer[batchNum][bufNum];

					// ���K��
					F32 value2 = (F32)( (value - mean) * sqrtVarianceInv );

					this->lpDScale[0] += this->lppBatchDOutputBuffer[batchNum][bufNum] * value2;
					this->lpDBias[0]  += this->lppBatchDOutputBuffer[batchNum][bufNum];
				}
			}

			// ���ςƕ��U���X�V
			this->layerData.lpMean[0]     = (F32)((1.0 - factor) * this->layerData.lpMean[0]     + factor * this->lpTmpMean[0]);
			this->layerData.lpVariance[0] = (F32)((1.0 - factor) * this->layerData.lpVariance[0] + factor * variance);
		}

		// �X�P�[���ƃo�C�A�X���X�V
		if(this->layerData.m_pOptimizer_scale)
			this->layerData.m_pOptimizer_scale->UpdateParameter(&this->layerData.lpScale[0], &this->lpDScale[0]);
		if(this->layerData.m_pOptimizer_bias)
			this->layerData.m_pOptimizer_bias->UpdateParameter(&this->layerData.lpBias[0], &this->lpDBias[0]);

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
