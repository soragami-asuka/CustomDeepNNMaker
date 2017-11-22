//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A������
// GPU�����p
//======================================
#include"stdafx.h"

#include"Reshape_SquaresZeroSideLeftTop_DATA.hpp"
#include"Reshape_SquaresZeroSideLeftTop_FUNC.hpp"
#include"Reshape_SquaresZeroSideLeftTop_Base.h"

#include"Reshape_SquaresZeroSideLeftTop_GPU.cuh"
#include"Reshape_SquaresZeroSideLeftTop_LayerData_GPU.cuh"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** �R���X�g���N�^ */
	Reshape_SquaresZeroSideLeftTop_GPU::Reshape_SquaresZeroSideLeftTop_GPU(Gravisbell::GUID guid, Reshape_SquaresZeroSideLeftTop_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct)
		:	Reshape_SquaresZeroSideLeftTop_Base				(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< ���C���[�f�[�^ */
		,	m_lpInputBuffer				(NULL)		/**< ���Z���̓��̓f�[�^ */
		,	m_lpDOutputBuffer			(NULL)		/**< �o�͌덷�f�[�^ */
	{
	}
	/** �f�X�g���N�^ */
	Reshape_SquaresZeroSideLeftTop_GPU::~Reshape_SquaresZeroSideLeftTop_GPU()
	{
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 Reshape_SquaresZeroSideLeftTop_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode Reshape_SquaresZeroSideLeftTop_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	Reshape_SquaresZeroSideLeftTop_LayerData_Base& Reshape_SquaresZeroSideLeftTop_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const Reshape_SquaresZeroSideLeftTop_LayerData_Base& Reshape_SquaresZeroSideLeftTop_GPU::GetLayerData()const
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
	ErrorCode Reshape_SquaresZeroSideLeftTop_GPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// �o�͌덷�o�b�t�@
		this->m_lpDOutputBuffer_h.resize(this->GetBatchSize() * this->outputBufferCount);
		this->m_lppDOutputBuffer.resize(this->GetBatchSize());
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			this->m_lppDOutputBuffer[batchNum] = &this->m_lpDOutputBuffer_h[batchNum * this->outputBufferCount];

		// ���͌덷�o�b�t�@
		this->m_lpDInputBuffer_h.resize(this->GetBatchSize() * this->inputBufferCount);
		this->m_lppDInputBuffer.resize(this->GetBatchSize());
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			this->m_lppDInputBuffer[batchNum] = &this->m_lpDInputBuffer_h[batchNum * this->inputBufferCount];


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode Reshape_SquaresZeroSideLeftTop_GPU::PreProcessCalculate()
	{
		// ���̓o�b�t�@�����m�F
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// �o�̓o�b�t�@�����m�F
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// ���̓o�b�t�@�ۑ��p�̃A�h���X�z����쐬
		this->m_lpInputBuffer_h.resize(this->inputBufferCount * this->GetBatchSize());
		this->m_lppInputBuffer.resize(this->GetBatchSize(), NULL);
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->m_lppInputBuffer[batchNum] = &this->m_lpInputBuffer_h[batchNum * this->inputBufferCount];
		}

		// �o�̓o�b�t�@���쐬
		this->m_lpOutputBuffer.resize(this->GetBatchSize() * this->outputBufferCount);
		this->m_lpOutputBuffer_d.resize(this->GetBatchSize() * this->outputBufferCount);
		this->m_lppOutputBuffer.resize(this->GetBatchSize());
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			this->m_lppOutputBuffer[batchNum] = &this->m_lpOutputBuffer[batchNum * this->outputBufferCount];
		}

		// �ϊ��e�[�u�����쐬
		this->m_lpConvertTable.resize(this->GetOutputDataStruct().y * this->GetOutputDataStruct().x);
		this->m_lppConvertTable.resize(this->GetOutputDataStruct().y);
		for(U32 y=0; y<this->GetOutputDataStruct().y; y++)
		{
			this->m_lppConvertTable[y] = &this->m_lpConvertTable[this->GetOutputDataStruct().x*y];

			for(U32 x=0; x<this->GetOutputDataStruct().x; x++)
			{
				U32 value = x*y;

				this->m_lppConvertTable[y][x] = value;
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}



	/** ���[�v�̏���������.�f�[�^�Z�b�g�̎��s�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode Reshape_SquaresZeroSideLeftTop_GPU::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode Reshape_SquaresZeroSideLeftTop_GPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
	{
		this->m_lpInputBuffer = i_lpInputBuffer;
		cudaMemcpy(&this->m_lpInputBuffer_h[0], this->m_lpInputBuffer, sizeof(F32)*this->inputBufferCount*this->GetBatchSize(), cudaMemcpyDeviceToHost);

		// �o�̓o�b�t�@�ɕϊ�
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			for(U32 ch=0; ch<this->GetOutputDataStruct().ch; ch++)
			{
				for(U32 outputZ=0; outputZ<this->GetOutputDataStruct().z; outputZ++)
				{
					for(U32 outputY=0; outputY<this->GetOutputDataStruct().y; outputY++)
					{
						for(U32 outputX=0; outputX<this->GetOutputDataStruct().x; outputX++)
						{
							U32 inputX = this->m_lppConvertTable[outputY][outputX];

							U32 outputOffset  = this->GetOutputDataStruct().POSITION_TO_OFFSET(outputX, outputY, outputZ, ch);
							U32 inputOffset   = this->GetInputDataStruct().POSITION_TO_OFFSET(inputX,  0, 0, ch);

							this->m_lppOutputBuffer[batchNum][outputOffset] = this->m_lppInputBuffer[batchNum][inputOffset];
						}
					}
				}
			}
		}

		// �o�̓o�b�t�@���f�o�C�X�ɃR�s�[
		cudaMemcpy(thrust::raw_pointer_cast(&this->m_lpOutputBuffer_d[0]), &this->m_lpOutputBuffer[0], sizeof(F32)*this->outputBufferCount*this->GetBatchSize(), cudaMemcpyHostToDevice);

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �o�̓f�[�^�o�b�t�@���擾����.
		�z��̗v�f����GetOutputBufferCount�̖߂�l.
		@return �o�̓f�[�^�z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER Reshape_SquaresZeroSideLeftTop_GPU::GetOutputBuffer()const
	{
		return thrust::raw_pointer_cast(&this->m_lpOutputBuffer_d[0]);
	}
	/** �o�̓f�[�^�o�b�t�@���擾����.
		@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
		@return ���������ꍇ0 */
	ErrorCode Reshape_SquaresZeroSideLeftTop_GPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
	{
		if(o_lpOutputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 outputBufferCount = this->GetOutputBufferCount();

		cudaMemcpy(o_lpOutputBuffer, this->GetOutputBuffer(), sizeof(F32)*outputBufferCount*this->GetBatchSize(), cudaMemcpyDeviceToHost);

		return ErrorCode::ERROR_CODE_NONE;
	}


	//================================
	// �w�K����
	//================================
	/** ���͌덷�v�Z�������s����.�w�K�����ɓ��͌덷���擾�������ꍇ�Ɏg�p����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	o_lppDInputBuffer	���͌덷�����i�[�惌�C���[.	[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v�Ȕz���[GetOutputDataCount()]�z��
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode Reshape_SquaresZeroSideLeftTop_GPU::CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lpDOutputBuffer)
	{
		// ���͌덷�v�Z
		this->m_lpDOutputBuffer = i_lpDOutputBuffer;
		this->m_lpDInputBuffer  = o_lppDInputBuffer;
		if(this->m_lpDOutputBuffer && this->m_lpDInputBuffer)
		{
			// �o�͌덷�o�b�t�@���z�X�g�ɃR�s�[
			cudaMemcpy(&this->m_lpDOutputBuffer_h[0], i_lpDOutputBuffer, sizeof(F32)*this->outputBufferCount*this->GetBatchSize(), cudaMemcpyDeviceToHost);

			// ���͌덷��������
			memset(&this->m_lpDInputBuffer_h[0], 0, sizeof(F32)*this->GetBatchSize()*this->inputBufferCount);

			// ���͌덷�v�Z
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				for(U32 ch=0; ch<this->GetOutputDataStruct().ch; ch++)
				{
					for(U32 outputZ=0; outputZ<this->GetOutputDataStruct().z; outputZ++)
					{
						for(U32 outputY=0; outputY<this->GetOutputDataStruct().y; outputY++)
						{
							for(U32 outputX=0; outputX<this->GetOutputDataStruct().x; outputX++)
							{
								U32 inputX = this->m_lppConvertTable[outputY][outputX];

								U32 outputOffset  = this->GetOutputDataStruct().POSITION_TO_OFFSET(outputX, outputY, outputZ, ch);
								U32 inputOffset   = this->GetInputDataStruct().POSITION_TO_OFFSET(inputX,  0, 0, ch);

								this->m_lppDInputBuffer[batchNum][inputOffset] += this->m_lppDOutputBuffer[batchNum][outputOffset];
							}
						}
					}
				}
			}

			// ���͌덷���f�o�C�X�ɃR�s�[
			cudaMemcpy(m_lpDInputBuffer, &this->m_lpDInputBuffer_h[0], sizeof(F32)*this->inputBufferCount*this->GetBatchSize(), cudaMemcpyHostToDevice);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode Reshape_SquaresZeroSideLeftTop_GPU::Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lpDOutputBuffer)
	{
		return this->CalculateDInput(o_lppDInputBuffer, i_lpDOutputBuffer);
	}


	/** �w�K�������擾����.
		�z��̗v�f����[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]
		@return	�덷�����z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER Reshape_SquaresZeroSideLeftTop_GPU::GetDInputBuffer()const
	{
		return this->m_lpDOutputBuffer;
	}
	/** �w�K�������擾����.
		@param lpDInputBuffer	�w�K�������i�[����z��.[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̔z�񂪕K�v */
	ErrorCode Reshape_SquaresZeroSideLeftTop_GPU::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
	{
		if(o_lpDInputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 inputBufferCount = this->GetInputBufferCount();

		cudaMemcpy(o_lpDInputBuffer, this->GetDInputBuffer(), sizeof(F32)*inputBufferCount*batchSize, cudaMemcpyDeviceToHost);

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
