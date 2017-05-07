//==================================
// �j���[�����l�b�g���[�N�̃��C���[�Ǘ��p��Utiltiy
// ���C�u�����Ƃ��Ďg���Ԃ͗L��.
// �c�[������͏����\��
//==================================
#include"stdafx.h"

#include"NeuralNetworkLayer.h"

#include<boost/uuid/uuid_generators.hpp>
#include<boost/foreach.hpp>

namespace Gravisbell {
namespace Utility {
namespace NeuralNetworkLayer {


/** ���C���[DLL�Ǘ��N���X�̍쐬 */
Layer::NeuralNetwork::ILayerDLLManager* CreateLayerDLLManagerCPU(const boost::filesystem::wpath& libraryDirPath)
{
	// DLL�Ǘ��N���X���쐬
	Layer::NeuralNetwork::ILayerDLLManager* pDLLManager = Layer::NeuralNetwork::CreateLayerDLLManagerCPU();

	BOOST_FOREACH(const boost::filesystem::wpath& path, std::make_pair(boost::filesystem::directory_iterator(libraryDirPath),
                                                    boost::filesystem::directory_iterator()))
	{
		if(path.stem().wstring().find(L"Gravisbell.Layer.NeuralNetwork.") == 0 && path.extension().wstring()==L".dll")
		{
			pDLLManager->ReadLayerDLL(path.generic_wstring().c_str());
		}
    }
	return pDLLManager;
}
Layer::NeuralNetwork::ILayerDLLManager* CreateLayerDLLManagerGPU(const boost::filesystem::wpath& libraryDirPath)
{
	// DLL�Ǘ��N���X���쐬
	Layer::NeuralNetwork::ILayerDLLManager* pDLLManager = Layer::NeuralNetwork::CreateLayerDLLManagerGPU();

	BOOST_FOREACH(const boost::filesystem::wpath& path, std::make_pair(boost::filesystem::directory_iterator(libraryDirPath),
                                                    boost::filesystem::directory_iterator()))
	{
		if(path.stem().wstring().find(L"Gravisbell.Layer.NeuralNetwork.") == 0 && path.extension().wstring()==L".dll")
		{
			pDLLManager->ReadLayerDLL(path.generic_wstring().c_str());
		}
    }
	return pDLLManager;
}


/** ���C���[�f�[�^���쐬 */
Layer::NeuralNetwork::INNLayerConnectData* CreateNeuralNetwork(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct)
{
	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(Gravisbell::GUID(0x1c38e21f, 0x6f01, 0x41b2, 0xb4, 0x0e, 0x7f, 0x67, 0x26, 0x7a, 0x36, 0x92));
	if(pLayerDLL == NULL)
		return NULL;

	// �ݒ�̍쐬
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// ���C���[�̍쐬
	Layer::NeuralNetwork::INNLayerData* pLayer = pLayerDLL->CreateLayerData(*pConfig, inputDataStruct);
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	// �L���X�g
	Layer::NeuralNetwork::INNLayerConnectData* pNeuralNetwork = dynamic_cast<Layer::NeuralNetwork::INNLayerConnectData*>(pLayer);
	if(pNeuralNetwork == NULL)
	{
		delete pLayer;
		return NULL;
	}

	return pNeuralNetwork;
}
Layer::NeuralNetwork::INNLayerData* CreateConvolutionLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, Vector3D<S32> filterSize, U32 outputChannelCount, Vector3D<S32> stride, Vector3D<S32> paddingSize)
{
	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(Gravisbell::GUID(0xf6662e0e, 0x1ca4, 0x4d59, 0xac, 0xca, 0xca, 0xc2, 0x9a, 0x16, 0xc0, 0xaa));
	if(pLayerDLL == NULL)
		return NULL;

	// �ݒ�̍쐬
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;
	// �t�B���^�T�C�Y
	{
		SettingData::Standard::IItem_Vector3D_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Vector3D_Int*>(pConfig->GetItemByID(L"FilterSize"));
		pItem->SetValue(filterSize);
	}
	// �o�̓`�����l����
	{
		SettingData::Standard::IItem_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Int*>(pConfig->GetItemByID(L"Output_Channel"));
		pItem->SetValue(outputChannelCount);
	}
	// �t�B���^�ړ���
	{
		SettingData::Standard::IItem_Vector3D_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Vector3D_Int*>(pConfig->GetItemByID(L"Stride"));
		pItem->SetValue(stride);
	}
	// �p�f�B���O�T�C�Y
	{
		SettingData::Standard::IItem_Vector3D_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Vector3D_Int*>(pConfig->GetItemByID(L"Padding"));
		pItem->SetValue(paddingSize);
	}
	// �p�f�B���O���
	{
		SettingData::Standard::IItem_Enum* pItem = dynamic_cast<SettingData::Standard::IItem_Enum*>(pConfig->GetItemByID(L"PaddingType"));
		pItem->SetValue(L"zero");
	}

	// ���C���[�̍쐬
	Layer::NeuralNetwork::INNLayerData* pLayer = pLayerDLL->CreateLayerData(*pConfig, inputDataStruct);
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	return pLayer;
}
Layer::NeuralNetwork::INNLayerData* CreateFullyConnectLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, U32 neuronCount)
{
	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(Gravisbell::GUID(0x14cc33f4, 0x8cd3, 0x4686, 0x9c, 0x48, 0xef, 0x45, 0x2b, 0xa5, 0xd2, 0x02));
	if(pLayerDLL == NULL)
		return NULL;

	// �ݒ�̍쐬
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;
	// �j���[������
	{
		SettingData::Standard::IItem_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Int*>(pConfig->GetItemByID(L"NeuronCount"));
		pItem->SetValue(neuronCount);
	}

	// ���C���[�̍쐬
	Layer::NeuralNetwork::INNLayerData* pLayer = pLayerDLL->CreateLayerData(*pConfig, inputDataStruct);
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	return pLayer;
}
Layer::NeuralNetwork::INNLayerData* CreateActivationLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, const std::wstring activationType)
{
	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(Gravisbell::GUID(0x99904134, 0x83b7, 0x4502, 0xa0, 0xca, 0x72, 0x8a, 0x2c, 0x9d, 0x80, 0xc7));
	if(pLayerDLL == NULL)
		return NULL;

	// �ݒ�̍쐬
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;
	// �������֐����
	{
		SettingData::Standard::IItem_Enum* pItem = dynamic_cast<SettingData::Standard::IItem_Enum*>(pConfig->GetItemByID(L"ActivationType"));
		pItem->SetValue(activationType.c_str());
	}

	// ���C���[�̍쐬
	Layer::NeuralNetwork::INNLayerData* pLayer = pLayerDLL->CreateLayerData(*pConfig, inputDataStruct);
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	return pLayer;
}
Layer::NeuralNetwork::INNLayerData* CreateDropoutLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, F32 rate)
{
	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(Gravisbell::GUID(0x298243e4, 0x2111, 0x474f, 0xa8, 0xf4, 0x35, 0xbd, 0xc8, 0x76, 0x45, 0x88));
	if(pLayerDLL == NULL)
		return NULL;

	// �ݒ�̍쐬
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;
	// �������֐����
	{
		SettingData::Standard::IItem_Float* pItem = dynamic_cast<SettingData::Standard::IItem_Float*>(pConfig->GetItemByID(L"Rate"));
		pItem->SetValue(rate);
	}

	// ���C���[�̍쐬
	Layer::NeuralNetwork::INNLayerData* pLayer = pLayerDLL->CreateLayerData(*pConfig, inputDataStruct);
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	return pLayer;
}
Layer::NeuralNetwork::INNLayerData* CreatePoolingLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, Vector3D<S32> filterSize)
{
	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(Gravisbell::GUID(0xeb80e0d0, 0x9d5a, 0x4ed1, 0xa8, 0x0d, 0xa1, 0x66, 0x7d, 0xe0, 0xc8, 0x90));
	if(pLayerDLL == NULL)
		return NULL;

	// �ݒ�̍쐬
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;
	// �t�B���^�T�C�Y
	{
		SettingData::Standard::IItem_Vector3D_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Vector3D_Int*>(pConfig->GetItemByID(L"FilterSize"));
		pItem->SetValue(filterSize);
	}

	// ���C���[�̍쐬
	Layer::NeuralNetwork::INNLayerData* pLayer = pLayerDLL->CreateLayerData(*pConfig, inputDataStruct);
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	return pLayer;
}
Layer::NeuralNetwork::INNLayerData* CreateBatchNormalizationLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct)
{
	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(Gravisbell::GUID(0xacd11a5a, 0xbfb5, 0x4951, 0x83, 0x82, 0x1d, 0xe8, 0x9d, 0xfa, 0x96, 0xa8));
	if(pLayerDLL == NULL)
		return NULL;

	// �ݒ�̍쐬
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;
	
	// ���C���[�̍쐬
	Layer::NeuralNetwork::INNLayerData* pLayer = pLayerDLL->CreateLayerData(*pConfig, inputDataStruct);
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	return pLayer;
}


/** ���C���[���l�b�g���[�N�̖����ɒǉ�����.GUID�͎������蓖��.���̓f�[�^�\���A�ŏIGUID���X�V����. */
Gravisbell::ErrorCode AddLayerToNetworkLast( Layer::NeuralNetwork::INNLayerConnectData& neuralNetwork, std::list<Layer::ILayerData*>& lppLayerData, Gravisbell::IODataStruct& inputDataStruct, Gravisbell::GUID& lastLayerGUID, Layer::NeuralNetwork::INNLayerData* pAddlayer)
{
	// GUID����
	Gravisbell::GUID guid = boost::uuids::random_generator()().data;

	lppLayerData.push_back(pAddlayer);
	neuralNetwork.AddLayer(guid, pAddlayer);

	// �ڑ�
	neuralNetwork.AddInputLayerToLayer(guid, lastLayerGUID);

	// ���݃��C���[�𒼑O���C���[�ɕύX
	inputDataStruct = pAddlayer->GetOutputDataStruct();
	lastLayerGUID = guid;

	return Gravisbell::ErrorCode::ERROR_CODE_NONE;
}


}	// NeuralNetworkLayer
}	// Utility
}	// Gravisbell
