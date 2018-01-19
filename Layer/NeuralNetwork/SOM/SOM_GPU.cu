//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// GPU�����p
//======================================
#include"stdafx.h"

#include"SOM_DATA.hpp"
#include"SOM_FUNC.hpp"
#include"SOM_Base.h"

#include"SOM_GPU.cuh"
#include"SOM_LayerData_GPU.cuh"


using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

#define BLOCK_SIZE	(16)

namespace
{
}

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

#define CODE_MATCH_RATE	(L"SOM_MATCH_RATE")

	/** �R���X�g���N�^ */
	SOM_GPU::SOM_GPU(Gravisbell::GUID guid, SOM_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	SOM_Base				(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< ���C���[�f�[�^ */
		,	inputBufferCount				(0)		/**< ���̓o�b�t�@�� */
		,	unitCount						(0)		/**< ���j�b�g�� */
		,	outputBufferCount				(0)		/**< �o�̓o�b�t�@�� */
		,	temporaryMemoryManager			(i_temporaryMemoryManager)
	{
		cublasCreate(&cublasHandle);
	}
	/** �f�X�g���N�^ */
	SOM_GPU::~SOM_GPU()
	{
		cublasDestroy(cublasHandle);
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 SOM_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode SOM_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	SOM_LayerData_Base& SOM_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const SOM_LayerData_Base& SOM_GPU::GetLayerData()const
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
	ErrorCode SOM_GPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// �p�����[�^�̕ω��ʃo�b�t�@
		this->lpDUnit.resize(this->unitCount * this->inputBufferCount);

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode SOM_GPU::PreProcessCalculate()
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


		// �e���j�b�g�̍��W���v�Z����
		thrust::host_vector<F32> lpTmpUnitPos(this->unitCount * this->layerData.layerStructure.DimensionCount);
		for(U32 unitNo=0; unitNo<this->unitCount; unitNo++)
		{
			U32 offset = unitNo * this->layerData.layerStructure.DimensionCount;

			U32 tmpNo = unitNo;
			for(U32 dimNo=0; dimNo<(U32)this->layerData.layerStructure.DimensionCount; dimNo++)
			{
				U32 pos = tmpNo % this->layerData.layerStructure.ResolutionCount;
				tmpNo /= this->layerData.layerStructure.ResolutionCount;

				lpTmpUnitPos[offset + dimNo] = (F32)pos / this->layerData.layerStructure.ResolutionCount;
			}
		}
		this->lpUnitPos = lpTmpUnitPos;


		// �ꎞ�o�b�t�@�̃T�C�Y��ݒ�
		this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), CODE_MATCH_RATE, sizeof(F32)*this->unitCount*this->layerData.layerStructure.DimensionCount);


		return ErrorCode::ERROR_CODE_NONE;
	}

	
	/** ���[�v�̏���������.�f�[�^�Z�b�g�̎��s�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode SOM_GPU::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}



	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode SOM_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// BMU(Best Matching Unit)�𒲂ׂ�
		{
			// �o�b�t�@���m��
			F32* lpTmpMatchRate = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), CODE_MATCH_RATE);

			// ��v�����v�Z
			{
				// C = aAB + bC;

				// CUBLAS��
				// 0, 4,  8
				// 1, 5,  9
				// 2, 6, 10
				// 3, 7, 11
				// �̂悤�ɏc�����ɃC���f�b�N�X���i�ލs��ō\������Ă���


				F32 alpha = 1.0f;
				F32 beta  = 0.0f;

				cublasSgemm(
					this->cublasHandle,
					CUBLAS_OP_T,
					CUBLAS_OP_N,
					this->unitCount,	// �s��A�̍s��
					this->GetBatchSize(),	// �s��B�̗�
					this->inputBufferCount,	// �s��A�̗�,�s��B�̍s��
					&alpha,
					thrust::raw_pointer_cast(&this->layerData.lpUnitData[0]),	// �s��A
					this->inputBufferCount,										// �s��A�̓]�u�O�̍s��
					i_lppInputBuffer,											// �s��B
					this->inputBufferCount,										// �s��B�̓]�u�O�̍s��
					&beta,
					&lpTmpMatchRate[0],
					this->outputBufferCount);
			}

			// �ő�l�����߂�
			{
				// ��Βl�̍ő�l�����߂�

			}

			// ���j�b�g���W�����߂�

			// �o�b�t�@���
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), CODE_MATCH_RATE);
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
	ErrorCode SOM_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		cudaMemset(o_lppDInputBuffer, 0, sizeof(F32)*this->inputBufferCount*this->GetBatchSize());

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode SOM_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
