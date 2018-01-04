//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A������
// GPU�����p
//======================================
#include"stdafx.h"

#include"MergeInput_DATA.hpp"
#include"MergeInput_FUNC.hpp"
#include"MergeInput_Base.h"

#include"MergeInput_GPU.cuh"
#include"MergeInput_LayerData_GPU.cuh"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** �R���X�g���N�^ */
	MergeInput_GPU::MergeInput_GPU(Gravisbell::GUID guid, MergeInput_LayerData_GPU& i_layerData, const std::vector<IODataStruct>& i_lpInputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	MergeInput_Base					(guid, i_lpInputDataStruct, i_layerData.GetOutputDataStruct(&i_lpInputDataStruct[0], (U32)i_lpInputDataStruct.size()))
		,	layerData						(i_layerData)	/**< ���C���[�f�[�^ */
		,	outputBufferCount				(0)				/**< �o�̓o�b�t�@�� */
	{
	}
	/** �f�X�g���N�^ */
	MergeInput_GPU::~MergeInput_GPU()
	{
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 MergeInput_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode MergeInput_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	MergeInput_LayerData_Base& MergeInput_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const MergeInput_LayerData_Base& MergeInput_GPU::GetLayerData()const
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
	ErrorCode MergeInput_GPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// �o�͌덷�o�b�t�@�̔z����쐬
		this->m_lppDInputBuffer.resize(this->GetInputDataCount());

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode MergeInput_GPU::PreProcessCalculate()
	{
		// ���̓o�b�t�@�����m�F
		this->lpInputBufferCount.resize(this->GetInputDataCount());
		for(U32 inputNum=0; inputNum<this->lpInputBufferCount.size(); inputNum++)
		{
			this->lpInputBufferCount[inputNum] = this->GetInputBufferCount(inputNum);
			if(this->lpInputBufferCount[inputNum] == 0)
				return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;
		}

		// �o�̓o�b�t�@�����m�F
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���[�v�̏���������.�f�[�^�Z�b�g�̎��s�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode MergeInput_GPU::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode MergeInput_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		switch(this->layerData.layerStructure.mergeDirection)
		{
		case MergeInput::LayerStructure::mergeDirection_ch:
			{
				for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
				{
					U32 offset = 0;
					for(U32 inputNum=0; inputNum<this->lpInputBufferCount.size(); inputNum++)
					{
						cudaError_t err = cudaMemcpyAsync(
							&o_lppOutputBuffer[batchNum*this->outputBufferCount + offset],
							&i_lppInputBuffer[inputNum][batchNum*this->lpInputBufferCount[inputNum]],
							sizeof(F32) * this->lpInputBufferCount[inputNum],
							cudaMemcpyDeviceToDevice);
						if(err != 0)
							return ErrorCode::ERROR_CODE_CUDA_CALCULATE;

						offset += this->lpInputBufferCount[inputNum];
					}
				}
				cudaThreadSynchronize();
			}
			break;
		case MergeInput::LayerStructure::mergeDirection_x:
			break;
		case MergeInput::LayerStructure::mergeDirection_y:
			break;
		case MergeInput::LayerStructure::mergeDirection_z:
			break;
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
	ErrorCode MergeInput_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		if(o_lppDInputBuffer)
		{
			// ���͌덷�o�b�t�@�̃A�h���X��z��Ɋi�[
			for(U32 inputNum=0; inputNum<this->GetInputDataCount(); inputNum++)
			{
				this->m_lppDInputBuffer[inputNum] = o_lppDInputBuffer[inputNum];
			}

			switch(this->layerData.layerStructure.mergeDirection)
			{
			case MergeInput::LayerStructure::mergeDirection_ch:
				{
					for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
					{
						U32 offset = 0;
						for(U32 inputNum=0; inputNum<this->lpInputBufferCount.size(); inputNum++)
						{
							cudaError_t err = cudaMemcpyAsync(
								&this->m_lppDInputBuffer[inputNum][batchNum*this->lpInputBufferCount[inputNum]],
								&i_lppDOutputBuffer[batchNum*this->outputBufferCount + offset],
								sizeof(F32) * this->lpInputBufferCount[inputNum],
								cudaMemcpyDeviceToDevice);
							if(err != 0)
								return ErrorCode::ERROR_CODE_CUDA_CALCULATE;

							offset += this->lpInputBufferCount[inputNum];
						}
					}
					cudaThreadSynchronize();
				}
				break;
			case MergeInput::LayerStructure::mergeDirection_x:
				break;
			case MergeInput::LayerStructure::mergeDirection_y:
				break;
			case MergeInput::LayerStructure::mergeDirection_z:
				break;
			}
		}
		

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode MergeInput_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
