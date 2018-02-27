//======================================
// �o�b�`���K�����C���[
// GPU�����p
//======================================
#include"stdafx.h"

#include"Normalization_Scale_DATA.hpp"
#include"Normalization_Scale_FUNC.hpp"
#include"Normalization_Scale_Base.h"

#include"Normalization_Scale_GPU.cuh"
#include"Normalization_Scale_LayerData_GPU.cuh"


using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** �R���X�g���N�^ */
	Normalization_Scale_GPU::Normalization_Scale_GPU(Gravisbell::GUID guid, Normalization_Scale_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	Normalization_Scale_Base	(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData					(i_layerData)	/**< ���C���[�f�[�^ */
		,	inputBufferCount			(0)				/**< ���̓o�b�t�@�� */
		,	outputBufferCount			(0)				/**< �o�̓o�b�t�@�� */
	{
	}
	/** �f�X�g���N�^ */
	Normalization_Scale_GPU::~Normalization_Scale_GPU()
	{
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 Normalization_Scale_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode Normalization_Scale_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	Normalization_Scale_LayerData_Base& Normalization_Scale_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const Normalization_Scale_LayerData_Base& Normalization_Scale_GPU::GetLayerData()const
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
	ErrorCode Normalization_Scale_GPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// �w�K�p�̕ϐ����쐬

		// ���͌덷�o�b�t�@
		this->m_lpDInputBuffer_h.resize(this->GetBatchSize() * this->inputBufferCount);
		this->m_lppDInputBuffer_h.resize(this->GetBatchSize());
		for(U32 i=0; i<this->GetBatchSize(); i++)
			this->m_lppDInputBuffer_h[i] = &this->m_lpDInputBuffer_h[i * this->inputBufferCount];

		// �o�͌덷�o�b�t�@
		this->m_lpDOutputBuffer_h.resize(this->GetBatchSize() * this->outputBufferCount);
		this->m_lppDOutputBuffer_h.resize(this->GetBatchSize());
		for(U32 i=0; i<this->GetBatchSize(); i++)
			this->m_lppDOutputBuffer_h[i] = &this->m_lpDOutputBuffer_h[i * this->outputBufferCount];

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode Normalization_Scale_GPU::PreProcessCalculate()
	{
		// ���ϒl�p�̃o�b�t�@���쐬
		this->lpTmpMean.resize(this->GetBatchSize());

		// ���̓o�b�t�@�����m�F
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// �o�̓o�b�t�@�����m�F
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// ���̓o�b�t�@���쐬
		this->m_lpInputBuffer_h.resize(this->inputBufferCount * this->GetBatchSize());
		this->m_lppInputBuffer_h.resize(this->GetBatchSize());
		for(U32 i=0; i<this->GetBatchSize(); i++)
			this->m_lppInputBuffer_h[i] = &this->m_lpInputBuffer_h[i * this->inputBufferCount];

		// �o�̓o�b�t�@���쐬
		this->m_lpOutputBuffer_h.resize(this->GetBatchSize() * this->outputBufferCount);
		this->m_lppOutputBuffer_h.resize(this->GetBatchSize());
		for(U32 i=0; i<this->GetBatchSize(); i++)
			this->m_lppOutputBuffer_h[i] = &this->m_lpOutputBuffer_h[i * this->outputBufferCount];


		return ErrorCode::ERROR_CODE_NONE;
	}

	
	/** ���[�v�̏���������.�f�[�^�Z�b�g�̎��s�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode Normalization_Scale_GPU::PreProcessLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode Normalization_Scale_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// ���̓o�b�t�@���z�X�g�ɃR�s�[
		cudaMemcpy(&this->m_lpInputBuffer_h[0], i_lppInputBuffer, sizeof(F32)*this->m_lpInputBuffer_h.size(), cudaMemcpyDeviceToHost);

		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			// ���ϒl�����߂�
			F32 ave = 0.0f;
			for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
			{
				ave += this->m_lppInputBuffer_h[batchNum][inputNum];
			}
			ave /= this->inputBufferCount;

			if(this->GetProcessType() == ProcessType::PROCESSTYPE_LEARN)
				this->lpTmpMean[batchNum] = ave;

			// �o�͂��v�Z����
			for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
			{
				this->m_lppOutputBuffer_h[batchNum][inputNum] = (this->m_lppInputBuffer_h[batchNum][inputNum] - ave) * this->layerData.scale + this->layerData.bias;
			}
		}

		// �o�̓o�b�t�@���f�o�C�X�ɃR�s�[
		cudaMemcpy(o_lppOutputBuffer, &this->m_lpOutputBuffer_h[0], sizeof(F32)*this->m_lpOutputBuffer_h.size(), cudaMemcpyHostToDevice);


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
	ErrorCode Normalization_Scale_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// �o�͌덷�o�b�t�@���z�X�g�ɃR�s�[
		cudaMemcpy(&this->m_lpDOutputBuffer_h[0], i_lppDOutputBuffer, sizeof(F32)*this->m_lpDOutputBuffer_h.size(), cudaMemcpyDeviceToHost);

		// ���͌덷�o�b�t�@�̃A�h���X��z��Ɋi�[
		this->m_lpDInputBuffer_d = o_lppDInputBuffer;
		if(o_lppDInputBuffer)
		{
			// ���͌덷���v�Z
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
				{
					this->m_lppDInputBuffer_h[batchNum][inputNum] = this->layerData.scale * (1.0f - 1.0f/this->inputBufferCount) * this->m_lppDOutputBuffer_h[batchNum][inputNum];
				}
			}

			// ���͌덷���f�o�C�X�ɃR�s�[
			cudaMemcpy(o_lppDInputBuffer, &this->m_lpDInputBuffer_h[0], sizeof(F32)*this->m_lpDInputBuffer_h.size(), cudaMemcpyHostToDevice);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode Normalization_Scale_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		Gravisbell::ErrorCode errCode = this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
		if(errCode != ErrorCode::ERROR_CODE_NONE)
			return errCode;

		// �X�P�[���ƃo�C�A�X�̕ω��ʂ��v�Z
		F32 dScale = 0.0f;
		F32 dBias  = 0.0f;

		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			F32 ave = this->lpTmpMean[batchNum];

			for(U32 inputNum=0; inputNum<this->inputBufferCount; inputNum++)
			{
				dScale += this->m_lppDOutputBuffer_h[batchNum][inputNum] * (this->m_lppInputBuffer_h[batchNum][inputNum] - ave);
				dBias  += this->m_lppDOutputBuffer_h[batchNum][inputNum];
			}
		}

		// �X�P�[���ƃo�C�A�X���X�V
		if(this->layerData.m_pOptimizer_scale)
			this->layerData.m_pOptimizer_scale->UpdateParameter(&this->layerData.scale, &dScale);
		if(this->layerData.m_pOptimizer_bias)
			this->layerData.m_pOptimizer_bias->UpdateParameter(&this->layerData.bias, &dBias);

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
