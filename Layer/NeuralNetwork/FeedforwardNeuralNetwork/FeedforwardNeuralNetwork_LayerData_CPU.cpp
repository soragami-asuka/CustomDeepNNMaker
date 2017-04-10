//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̏������C���[�̃f�[�^
// �����̃��C���[�����A��������
//======================================
#include"stdafx.h"

#include"FeedforwardNeuralNetwork_FUNC.hpp"
#include"FeedforwardNeuralNetwork_LayerData_Base.h"
#include"FeedforwardNeuralNetwork_CPU.h"

using namespace Gravisbell;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class FeedforwardNeuralNetwork_LayerData_CPU : public FeedforwardNeuralNetwork_LayerData_Base
	{	
		//====================================
		// �R���X�g���N�^/�f�X�g���N�^
		//====================================
	public:
		/** �R���X�g���N�^ */
		FeedforwardNeuralNetwork_LayerData_CPU(const ILayerDLLManager& i_layerDLLManager, const Gravisbell::GUID& guid)
			:	FeedforwardNeuralNetwork_LayerData_Base(i_layerDLLManager, guid)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~FeedforwardNeuralNetwork_LayerData_CPU()
		{
		}

		//===========================
		// ���C���[�쐬
		//===========================
		/** ���C���[���쐬����.
			@param guid	�V�K�������郌�C���[��GUID. */
		INNLayer* CreateLayer(const Gravisbell::GUID& guid)
		{
			FeedforwardNeuralNetwork_Base* pNeuralNetwork = new FeedforwardNeuralNetwork_CPU(guid, *this);

			// �j���[�����l�b�g���[�N�Ƀ��C���[��ǉ�
			ErrorCode err = AddConnectionLayersToNeuralNetwork(*pNeuralNetwork);
			if(err != ErrorCode::ERROR_CODE_NONE)
			{
				delete pNeuralNetwork;
				return NULL;
			}

			return pNeuralNetwork;
		}
	};


}	// NeuralNetwork
}	// Layer
}	// Gravisbell


/** Create a layer for CPU processing.
  * @param GUID of layer to create.
  */
EXPORT_API Gravisbell::Layer::NeuralNetwork::INNLayerData* CreateLayerDataCPU(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const Gravisbell::SettingData::Standard::IData& i_data, const Gravisbell::IODataStruct& i_inputDataStruct)
{
	// DLL�}�l�[�W����NULL�`�F�b�N
	if(pLayerDLLManager == NULL)
		return NULL;

	// �쐬
	Gravisbell::Layer::NeuralNetwork::FeedforwardNeuralNetwork_LayerData_CPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::FeedforwardNeuralNetwork_LayerData_CPU(*pLayerDLLManager, guid);
	if(pLayerData == NULL)
		return NULL;

	// ������
	Gravisbell::ErrorCode errCode = pLayerData->Initialize(i_data, i_inputDataStruct);
	if(errCode != Gravisbell::ErrorCode::ERROR_CODE_NONE)
	{
		delete pLayerData;
		return NULL;
	}

	return pLayerData;
}
EXPORT_API Gravisbell::Layer::NeuralNetwork::INNLayerData* CreateLayerDataCPUfromBuffer(const Gravisbell::Layer::NeuralNetwork::ILayerDLLManager* pLayerDLLManager, Gravisbell::GUID guid, const BYTE* i_lpBuffer, S32 i_bufferSize, S32& o_useBufferSize)
{
	// DLL�}�l�[�W����NULL�`�F�b�N
	if(pLayerDLLManager == NULL)
		return NULL;

	// �쐬
	Gravisbell::Layer::NeuralNetwork::FeedforwardNeuralNetwork_LayerData_CPU* pLayerData = new Gravisbell::Layer::NeuralNetwork::FeedforwardNeuralNetwork_LayerData_CPU(*pLayerDLLManager, guid);
	if(pLayerData == NULL)
		return NULL;

	// �ǂݎ��Ɏg�p����o�b�t�@�����擾
	U32 useBufferSize = pLayerData->GetUseBufferByteCount();
	if(useBufferSize >= (U32)i_bufferSize)
	{
		delete pLayerData;
		return NULL;
	}

	// ������
	Gravisbell::ErrorCode errCode = pLayerData->InitializeFromBuffer(i_lpBuffer, i_bufferSize);
	if(errCode != Gravisbell::ErrorCode::ERROR_CODE_NONE)
	{
		delete pLayerData;
		return NULL;
	}

	// �g�p�����o�b�t�@�ʂ��i�[
	o_useBufferSize = useBufferSize;

	return pLayerData;
}
