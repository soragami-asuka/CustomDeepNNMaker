//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̏������C���[
// �����̃��C���[�����A��������
// CPU����
//======================================
#include"stdafx.h"

#include"FeedforwardNeuralNetwork_Base.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class FeedforwardNeuralNetwork_CPU : public FeedforwardNeuralNetwork_Base
	{
		//====================================
		// �R���X�g���N�^/�f�X�g���N�^
		//====================================
	public:
		/** �R���X�g���N�^ */
		FeedforwardNeuralNetwork_CPU(const ILayerDLLManager& layerDLLManager, const Gravisbell::GUID& i_guid)
			:	FeedforwardNeuralNetwork_Base(layerDLLManager, i_guid)
		{
		}
		/** �R���X�g���N�^
			@param	i_inputGUID	���͐M���Ɋ��蓖�Ă�ꂽGUID.�����ō�邱�Ƃ��ł��Ȃ��̂ŊO���ō쐬���Ĉ����n��. */
		FeedforwardNeuralNetwork_CPU(const ILayerDLLManager& layerDLLManager, const Gravisbell::GUID& i_guid, const Gravisbell::GUID& i_inputLayerGUID)
			:	FeedforwardNeuralNetwork_Base(layerDLLManager, i_guid, i_inputLayerGUID)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~FeedforwardNeuralNetwork_CPU()
		{
		}

	public:
		/** ���C���[��ʂ̎擾.
			ELayerKind �̑g�ݍ��킹. */
		U32 GetLayerKind(void)const
		{
			return this->GetLayerKindBase() | Gravisbell::Layer::LAYER_KIND_CPU;
		}

	public:
		//====================================
		// ���o�̓o�b�t�@�֘A
		//====================================
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
				memcpy(o_lpOutputBuffer[batchNum], FeedforwardNeuralNetwork_Base::GetOutputBuffer()[batchNum], sizeof(F32)*outputBufferCount);
			}

			return ErrorCode::ERROR_CODE_NONE;
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
				memcpy(o_lpDInputBuffer[batchNum], FeedforwardNeuralNetwork_Base::GetDInputBuffer()[batchNum], sizeof(F32)*inputBufferCount);
			}

			return ErrorCode::ERROR_CODE_NONE;
		}
	};


}	// NeuralNetwork
}	// Layer
}	// Gravisbell


/** Create a layer for CPU processing.
	* @param GUID of layer to create.
	*/
EXPORT_API Gravisbell::Layer::NeuralNetwork::INNLayer* CreateLayerCPU(Gravisbell::GUID guid, const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager)
{
	if(pLayerDLLManager == NULL)
		return NULL;


	return new Gravisbell::Layer::NeuralNetwork::FeedforwardNeuralNetwork_CPU(*pLayerDLLManager, guid);
}