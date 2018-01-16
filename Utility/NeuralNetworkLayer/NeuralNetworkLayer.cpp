//==================================
// �j���[�����l�b�g���[�N�̃��C���[�Ǘ��p��Utiltiy
// ���C�u�����Ƃ��Ďg���Ԃ͗L��.
// �c�[������͏����\��
//==================================
#include"stdafx.h"

#include"Utility/NeuralNetworkLayer.h"
#include"Library/NeuralNetwork/LayerDLLManager.h"

#include<boost/uuid/uuid_generators.hpp>
#include<boost/foreach.hpp>


namespace Gravisbell {
namespace Utility {
namespace NeuralNetworkLayer {


/** ���C���[DLL�Ǘ��N���X�̍쐬 */
Layer::NeuralNetwork::ILayerDLLManager* CreateLayerDLLManagerCPU(const wchar_t i_libraryDirPath[])
{
	// DLL�Ǘ��N���X���쐬
	Layer::NeuralNetwork::ILayerDLLManager* pDLLManager = Layer::NeuralNetwork::CreateLayerDLLManagerCPU();

	BOOST_FOREACH(const boost::filesystem::wpath& path, std::make_pair(boost::filesystem::directory_iterator(i_libraryDirPath),
                                                    boost::filesystem::directory_iterator()))
	{
		if(path.stem().wstring().find(L"Gravisbell.Layer.NeuralNetwork.") == 0 && path.extension().wstring()==L".dll")
		{
			pDLLManager->ReadLayerDLL(path.generic_wstring().c_str());
		}
    }
	return pDLLManager;
}
Layer::NeuralNetwork::ILayerDLLManager* CreateLayerDLLManagerGPU(const wchar_t i_libraryDirPath[])
{
	// DLL�Ǘ��N���X���쐬
	Layer::NeuralNetwork::ILayerDLLManager* pDLLManager = Layer::NeuralNetwork::CreateLayerDLLManagerGPU();

	BOOST_FOREACH(const boost::filesystem::wpath& path, std::make_pair(boost::filesystem::directory_iterator(i_libraryDirPath),
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
Layer::Connect::ILayerConnectData* CreateNeuralNetwork(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager)
{
	const Gravisbell::GUID TYPE_CODE(0x1c38e21f, 0x6f01, 0x41b2, 0xb4, 0x0e, 0x7f, 0x67, 0x26, 0x7a, 0x36, 0x92);

	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// �ݒ�̍쐬
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// ���C���[�̍쐬
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	// �L���X�g
	Layer::Connect::ILayerConnectData* pNeuralNetwork = dynamic_cast<Layer::Connect::ILayerConnectData*>(pLayer);
	if(pNeuralNetwork == NULL)
	{
		delete pLayer;
		return NULL;
	}

	return pNeuralNetwork;
}
Layer::ILayerData* CreateConvolutionLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	U32 inputChannelCount, Vector3D<S32> filterSize, U32 outputChannelCount, Vector3D<S32> stride, Vector3D<S32> paddingSize,
	const wchar_t i_szInitializerID[])
{
	const Gravisbell::GUID TYPE_CODE(0xf6662e0e, 0x1ca4, 0x4d59, 0xac, 0xca, 0xca, 0xc2, 0x9a, 0x16, 0xc0, 0xaa);

	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// �ݒ�̍쐬
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;
	// ���̓`�����l����
	{
		SettingData::Standard::IItem_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Int*>(pConfig->GetItemByID(L"Input_Channel"));
		pItem->SetValue(inputChannelCount);
	}
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
	// ���������@
	{
		SettingData::Standard::IItem_String* pItem = dynamic_cast<SettingData::Standard::IItem_String*>(pConfig->GetItemByID(L"Initializer"));
		pItem->SetValue(i_szInitializerID);
	}

	// ���C���[�̍쐬
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	return pLayer;
}
Layer::ILayerData* CreateFullyConnectLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	U32 inputBufferCount, U32 neuronCount,
	const wchar_t i_szInitializerID[])
{
	const Gravisbell::GUID TYPE_CODE(0x14cc33f4, 0x8cd3, 0x4686, 0x9c, 0x48, 0xef, 0x45, 0x2b, 0xa5, 0xd2, 0x02);


	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// �ݒ�̍쐬
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;
	// ���̓o�b�t�@��
	{
		SettingData::Standard::IItem_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Int*>(pConfig->GetItemByID(L"InputBufferCount"));
		pItem->SetValue(inputBufferCount);
	}
	// �j���[������
	{
		SettingData::Standard::IItem_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Int*>(pConfig->GetItemByID(L"NeuronCount"));
		pItem->SetValue(neuronCount);
	}
	// ���������@
	{
		SettingData::Standard::IItem_String* pItem = dynamic_cast<SettingData::Standard::IItem_String*>(pConfig->GetItemByID(L"Initializer"));
		pItem->SetValue(i_szInitializerID);
	}

	// ���C���[�̍쐬
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	return pLayer;
}
Layer::ILayerData* CreateActivationLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	const wchar_t activationType[])
{
	const Gravisbell::GUID TYPE_CODE(0x99904134, 0x83b7, 0x4502, 0xa0, 0xca, 0x72, 0x8a, 0x2c, 0x9d, 0x80, 0xc7);


	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// �ݒ�̍쐬
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;
	// �������֐����
	{
		SettingData::Standard::IItem_Enum* pItem = dynamic_cast<SettingData::Standard::IItem_Enum*>(pConfig->GetItemByID(L"ActivationType"));
		pItem->SetValue(activationType);
	}

	// ���C���[�̍쐬
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	return pLayer;
}
Layer::ILayerData* CreateDropoutLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	F32 rate)
{
	const Gravisbell::GUID TYPE_CODE(0x298243e4, 0x2111, 0x474f, 0xa8, 0xf4, 0x35, 0xbd, 0xc8, 0x76, 0x45, 0x88);


	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
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
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	return pLayer;
}
/** �K�E�X�m�C�Y���C���[.
	@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
	@param	inputDataStruct		���̓f�[�^�\��.
	@param	average				�������闐���̕��ϒl
	@param	variance			�������闐���̕��U */
GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
Layer::ILayerData* CreateGaussianNoiseLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, F32 average, F32 variance)
{
	const Gravisbell::GUID TYPE_CODE(0xac27c912, 0xa11d, 0x4519, 0x81, 0xa0, 0x17, 0xc0, 0x78, 0xe4, 0x43, 0x1f);


	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// �ݒ�̍쐬
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// ����
	SettingData::Standard::IItem_Float* pItem_Average = dynamic_cast<SettingData::Standard::IItem_Float*>(pConfig->GetItemByID(L"Average"));
	if(pItem_Average)
	{
		pItem_Average->SetValue(average);
	}

	// ���U
	SettingData::Standard::IItem_Float* pItem_Variance = dynamic_cast<SettingData::Standard::IItem_Float*>(pConfig->GetItemByID(L"Variance"));
	if(pItem_Variance)
	{
		pItem_Variance->SetValue(variance);
	}

	// ���C���[�̍쐬
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	return pLayer;
}

/** �v�[�����O���C���[.
	@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
	@param	inputDataStruct		���̓f�[�^�\��.
	@param	filterSize			�v�[�����O��.
	@param	stride				�t�B���^�ړ���. */
Layer::ILayerData* CreatePoolingLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	Vector3D<S32> filterSize, Vector3D<S32> stride)
{
	const Gravisbell::GUID TYPE_CODE(0xeb80e0d0, 0x9d5a, 0x4ed1, 0xa8, 0x0d, 0xa1, 0x66, 0x7d, 0xe0, 0xc8, 0x90);


	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
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
	// �X�g���C�h
	{
		SettingData::Standard::IItem_Vector3D_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Vector3D_Int*>(pConfig->GetItemByID(L"Stride"));
		pItem->SetValue(stride);
	}

	// ���C���[�̍쐬
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	return pLayer;
}

/** �o�b�`���K�����C���[ */
Layer::ILayerData* CreateBatchNormalizationLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	U32 inputChannelCount)
{
	const Gravisbell::GUID TYPE_CODE(0xacd11a5a, 0xbfb5, 0x4951, 0x83, 0x82, 0x1d, 0xe8, 0x9d, 0xfa, 0x96, 0xa8);


	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// �ݒ�̍쐬
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// ���̓`�����l����
	{
		SettingData::Standard::IItem_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Int*>(pConfig->GetItemByID(L"InputChannelCount"));
		pItem->SetValue(inputChannelCount);
	}

	// ���C���[�̍쐬
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	return pLayer;
}

/** �o�b�`���K�����C���[(�`�����l����ʂȂ�)
	@param	layerDLLManager		���C���[DLL�Ǘ��N���X. */
Layer::ILayerData* CreateBatchNormalizationAllLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager)
{
	const Gravisbell::GUID TYPE_CODE(0x8aecb925, 0x8dcf, 0x4876, 0xba, 0x6a, 0x6a, 0xdb, 0xe2, 0x80, 0xd2, 0x85);


	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// �ݒ�̍쐬
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// ���C���[�̍쐬
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	return pLayer;
}

/** �X�P�[�����K�����C���[ */
Layer::ILayerData* CreateNormalizationScaleLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager)
{
	const Gravisbell::GUID TYPE_CODE(0xd8c0de15, 0x5445, 0x482d, 0xbb, 0xc9, 0x00, 0x26, 0xbf, 0xa9, 0x6a, 0xdd);

	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// �ݒ�̍쐬
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// ���C���[�̍쐬
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	return pLayer;
}

/** �L�敽�σv�[�����O���C���[
	@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
	@param	inputDataStruct		���̓f�[�^�\��. */
Layer::ILayerData* CreateGlobalAveragePoolingLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager)
{
	const Gravisbell::GUID TYPE_CODE(0xf405d6d7, 0x434c, 0x4ed2, 0x82, 0xc3, 0x5d, 0x7e, 0x49, 0xf4, 0x03, 0xdb);


	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// �ݒ�̍쐬
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;
	
	// ���C���[�̍쐬
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	return pLayer;
}

/** GAN�ɂ�����Discriminator�̏o�̓��C���[
	@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
	@param	inputDataStruct		���̓f�[�^�\��. */
Layer::ILayerData* CreateActivationDiscriminatorLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager)
{
	const Gravisbell::GUID TYPE_CODE(0x6694e58a, 0x954c, 0x4092, 0x86, 0xc9, 0x65, 0x3d, 0x2e, 0x12, 0x4e, 0x83);


	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// �ݒ�̍쐬
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;
	
	// ���C���[�̍쐬
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	return pLayer;
}

Layer::ILayerData* CreateUpSamplingLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	Vector3D<S32> upScale, bool paddingUseValue)
{
	const Gravisbell::GUID TYPE_CODE(0x14eee4a7, 0x1b26, 0x4651, 0x8e, 0xbf, 0xb1, 0x15, 0x6d, 0x62, 0xce, 0x1b);


	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// �ݒ�̍쐬
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// �g�嗦
	{
		SettingData::Standard::IItem_Vector3D_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Vector3D_Int*>(pConfig->GetItemByID(L"UpScale"));
		pItem->SetValue(upScale);
	}
	// �p�f�B���O���
	{
		SettingData::Standard::IItem_Enum* pItem = dynamic_cast<SettingData::Standard::IItem_Enum*>(pConfig->GetItemByID(L"PaddingType"));
		if(paddingUseValue)
		{
			pItem->SetValue(L"value");
		}
		else
		{
			pItem->SetValue(L"zero");
		}
	}

	// ���C���[�̍쐬
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	return pLayer;
}

Layer::ILayerData* CreateMergeInputLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager)
{
	const Gravisbell::GUID TYPE_CODE(0x53daec93, 0xdbdb, 0x4048, 0xbd, 0x5a, 0x40, 0x1d, 0xd0, 0x05, 0xc7, 0x4e);


	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// �ݒ�̍쐬
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;
	
	// ���C���[�̍쐬
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	return pLayer;
}

/** ���͌������C���[(���Z). ���͂��ꂽ���C���[�̒l�����Z����. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������.
	@param	layerDLLManager		���C���[DLL�Ǘ��N���X
	@param	i_mergeType			ch�����}�[�W��������@. */
Layer::ILayerData* CreateMergeAddLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	LayerMergeType i_mergeType)
{
	const Gravisbell::GUID TYPE_CODE(0x754f6bbf, 0x7931, 0x473e, 0xae, 0x82, 0x29, 0xe9, 0x99, 0xa3, 0x4b, 0x22);


	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// �ݒ�̍쐬
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// �p�f�B���O���
	{
		SettingData::Standard::IItem_Enum* pItem = dynamic_cast<SettingData::Standard::IItem_Enum*>(pConfig->GetItemByID(L"MergeType"));
		switch(i_mergeType)
		{
		case LayerMergeType::LYAERMERGETYPE_MAX:
			pItem->SetValue(L"max");
			break;
		case LayerMergeType::LYAERMERGETYPE_MIN:
			pItem->SetValue(L"min");
			break;
		case LayerMergeType::LYAERMERGETYPE_LAYER0:
			pItem->SetValue(L"layer0");
			break;
		}
	}

	// ���C���[�̍쐬
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	return pLayer;
}


/** ���͌������C���[(����). ���͂��ꂽ���C���[�̒l�̕��ς��Ƃ�. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������.
	@param	layerDLLManager		���C���[DLL�Ǘ��N���X
	@param	i_mergeType			ch�����}�[�W��������@. */
GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
Layer::ILayerData* CreateMergeAverageLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	LayerMergeType i_mergeType)
{
	const Gravisbell::GUID TYPE_CODE(0x4e993b4b, 0x9f7a, 0x4cef, 0xa4, 0xc4, 0x37, 0xb9, 0x16, 0xbf, 0xd9, 0xb2);


	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// �ݒ�̍쐬
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// �p�f�B���O���
	{
		SettingData::Standard::IItem_Enum* pItem = dynamic_cast<SettingData::Standard::IItem_Enum*>(pConfig->GetItemByID(L"MergeType"));
		switch(i_mergeType)
		{
		case LayerMergeType::LYAERMERGETYPE_MAX:
			pItem->SetValue(L"max");
			break;
		case LayerMergeType::LYAERMERGETYPE_MIN:
			pItem->SetValue(L"min");
			break;
		case LayerMergeType::LYAERMERGETYPE_LAYER0:
			pItem->SetValue(L"layer0");
			break;
		}
	}

	// ���C���[�̍쐬
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	return pLayer;
}



/** ���͌������C���[(�ő�l). ���͂��ꂽ���C���[�̍ő�l���Z�o����. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������.
	@param	layerDLLManager		���C���[DLL�Ǘ��N���X
	@param	i_mergeType			ch�����}�[�W��������@. */
GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
Layer::ILayerData* CreateMergeMaxLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	LayerMergeType i_mergeType)
{
	const Gravisbell::GUID TYPE_CODE(0x3f015946, 0x7e88, 0x4db0, 0x91, 0xbd, 0xf4, 0x01, 0x3f, 0x21, 0x90, 0xd4);


	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// �ݒ�̍쐬
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// �p�f�B���O���
	{
		SettingData::Standard::IItem_Enum* pItem = dynamic_cast<SettingData::Standard::IItem_Enum*>(pConfig->GetItemByID(L"MergeType"));
		switch(i_mergeType)
		{
		case LayerMergeType::LYAERMERGETYPE_MAX:
			pItem->SetValue(L"max");
			break;
		case LayerMergeType::LYAERMERGETYPE_MIN:
			pItem->SetValue(L"min");
			break;
		case LayerMergeType::LYAERMERGETYPE_LAYER0:
			pItem->SetValue(L"layer0");
			break;
		}
	}

	// ���C���[�̍쐬
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	return pLayer;
}


/** �`�����l�����o���C���[. ���͂��ꂽ���C���[�̓���`�����l���𒊏o����. ����/�o�̓f�[�^�\����X,Y,Z�͓����T�C�Y.
	@param	startChannelNo	�J�n�`�����l���ԍ�.
	@param	channelCount	���o�`�����l����. */
Layer::ILayerData* CreateChooseChannelLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	U32 startChannelNo, U32 channelCount)
{
	const Gravisbell::GUID TYPE_CODE(0x244824b3, 0xbcfc, 0x4655, 0xa9, 0x91, 0x0f, 0x61, 0x36, 0xd3, 0x7a, 0x34);

	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// �ݒ�̍쐬
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// �J�n�`�����l���ԍ�
	{
		SettingData::Standard::IItem_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Int*>(pConfig->GetItemByID(L"startChannelNo"));
		pItem->SetValue(startChannelNo);
	}
	// �o�̓`�����l����
	{
		SettingData::Standard::IItem_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Int*>(pConfig->GetItemByID(L"channelCount"));
		pItem->SetValue(channelCount);
	}

	// ���C���[�̍쐬
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	return pLayer;
}


/** �o�̓f�[�^�\���ϊ����C���[.
	@param	ch	CH��.
	@param	x	X��.
	@param	y	Y��.
	@param	z	Z��. */
Layer::ILayerData* CreateReshapeLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	U32 ch, U32 x, U32 y, U32 z)
{
	const Gravisbell::GUID TYPE_CODE(0xe78e7f59, 0xd4b3, 0x45a1, 0xae, 0xeb, 0x9f, 0x2a, 0x51, 0x55, 0x47, 0x3f);

	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// �ݒ�̍쐬
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// CH��
	{
		SettingData::Standard::IItem_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Int*>(pConfig->GetItemByID(L"ch"));
		pItem->SetValue(ch);
	}
	// X��
	{
		SettingData::Standard::IItem_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Int*>(pConfig->GetItemByID(L"x"));
		pItem->SetValue(x);
	}
	// Y��
	{
		SettingData::Standard::IItem_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Int*>(pConfig->GetItemByID(L"y"));
		pItem->SetValue(y);
	}
	// Z��
	{
		SettingData::Standard::IItem_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Int*>(pConfig->GetItemByID(L"z"));
		pItem->SetValue(z);
	}

	// ���C���[�̍쐬
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	return pLayer;
}
/** �o�̓f�[�^�\���ϊ����C���[.
	@param	outputDataStruct �o�̓f�[�^�\�� */
Layer::ILayerData* CreateReshapeLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	const IODataStruct& outputDataStruct)
{
	return CreateReshapeLayer(layerDLLManager, layerDataManager, outputDataStruct.ch, outputDataStruct.x, outputDataStruct.y, outputDataStruct.z);
}


/** X=0�𒆐S�Ƀ~���[������*/
Layer::ILayerData* CreateReshapeMirrorXLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager)
{
	const Gravisbell::GUID TYPE_CODE(0xdfca3f81, 0xc2f1, 0x4ac6, 0xb6, 0x18, 0x81, 0x66, 0x51, 0xad, 0xdb, 0x63);

	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// �ݒ�̍쐬
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// ���C���[�̍쐬
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	return pLayer;
}

/** X=0�𒆐S�ɕ���������. */
Layer::ILayerData* CreateReshapeSquareCenterCrossLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager)
{
	const Gravisbell::GUID TYPE_CODE(0x5c2729d1, 0x33eb, 0x45ef, 0xab, 0xa5, 0x0c, 0x36, 0xac, 0x22, 0xd0, 0xbc);

	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// �ݒ�̍쐬
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// ���C���[�̍쐬
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	return pLayer;
}

/** X=0�𒆐S�ɕ���������. */
Layer::ILayerData* CreateReshapeSquareZeroSideLeftTopLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
	Gravisbell::U32 x, Gravisbell::U32 y)
{
	const Gravisbell::GUID TYPE_CODE(0xf6d9c5da, 0xd583, 0x455b, 0x92, 0x54, 0x5a, 0xef, 0x3c, 0xa9, 0x02, 0x1b);

	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// �ݒ�̍쐬
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// X��
	{
		SettingData::Standard::IItem_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Int*>(pConfig->GetItemByID(L"x"));
		pItem->SetValue(x);
	}
	// Y��
	{
		SettingData::Standard::IItem_Int* pItem = dynamic_cast<SettingData::Standard::IItem_Int*>(pConfig->GetItemByID(L"y"));
		pItem->SetValue(y);
	}

	// ���C���[�̍쐬
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	return pLayer;
}



Layer::ILayerData* CreateResidualLayer(
	const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager)
{
	const Gravisbell::GUID TYPE_CODE(0x0519e7fa, 0x311d, 0x4a1d, 0xa6, 0x15, 0x95, 0x9a, 0xfd, 0xd0, 0x05, 0x26);


	// DLL�擾
	const Gravisbell::Layer::NeuralNetwork::ILayerDLL* pLayerDLL = layerDLLManager.GetLayerDLLByGUID(TYPE_CODE);
	if(pLayerDLL == NULL)
		return NULL;

	// �ݒ�̍쐬
	SettingData::Standard::IData* pConfig = pLayerDLL->CreateLayerStructureSetting();
	if(pConfig == NULL)
		return NULL;

	// ���C���[�̍쐬
	Layer::ILayerData* pLayer = layerDataManager.CreateLayerData(layerDLLManager, TYPE_CODE, boost::uuids::random_generator()().data, *pConfig);
	if(pLayer == NULL)
		return NULL;

	// �ݒ�����폜
	delete pConfig;

	return pLayer;
}



/** ���C���[���l�b�g���[�N�̖����ɒǉ�����.GUID�͎������蓖��.���̓f�[�^�\���A�ŏIGUID���X�V����. */
Gravisbell::ErrorCode AddLayerToNetworkLast(
	Layer::Connect::ILayerConnectData& neuralNetwork,
	Gravisbell::GUID& lastLayerGUID, Layer::ILayerData* pAddlayer, bool onLayerFix)
{
	if(pAddlayer)
	{
		// GUID����
		Gravisbell::GUID guid = boost::uuids::random_generator()().data;

		neuralNetwork.AddLayer(guid, pAddlayer, onLayerFix);

		// �ڑ�
		neuralNetwork.AddInputLayerToLayer(guid, lastLayerGUID);

		// ���݃��C���[�𒼑O���C���[�ɕύX
		lastLayerGUID = guid;

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}

	return Gravisbell::ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
}

Gravisbell::ErrorCode AddLayerToNetworkLast(
	Layer::Connect::ILayerConnectData& neuralNetwork,
	Gravisbell::GUID& lastLayerGUID, Layer::ILayerData* pAddLayer, bool onLayerFix,
	const Gravisbell::GUID lpInputLayerGUID[], U32 inputLayerCount)
{
	if(pAddLayer)
	{
		// GUID����
		Gravisbell::GUID guid = boost::uuids::random_generator()().data;

		neuralNetwork.AddLayer(guid, pAddLayer, onLayerFix);

		// �ڑ�
		for(U32 inputNum=0; inputNum<inputLayerCount; inputNum++)
			neuralNetwork.AddInputLayerToLayer(guid, lpInputLayerGUID[inputNum]);

		// ���݃��C���[�𒼑O���C���[�ɕύX
		lastLayerGUID = guid;

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}

	return Gravisbell::ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
}


/** �j���[�����l�b�g���[�N���o�C�i���t�@�C���ɕۑ����� */
Gravisbell::ErrorCode WriteNetworkToBinaryFile(const Layer::ILayerData& neuralNetwork, const wchar_t i_filePath[])
{
	boost::filesystem::path filePath = i_filePath;

	// �o�b�t�@��p�ӂ���
	std::vector<BYTE> lpBuffer;
	S32 writeByteCount = 0;
	lpBuffer.resize(sizeof(Gravisbell::GUID) + neuralNetwork.GetUseBufferByteCount());

	// ���C���[��ʂ���������
	Gravisbell::GUID typeCode = neuralNetwork.GetLayerCode();
	memcpy(&lpBuffer[writeByteCount], &typeCode, sizeof(Gravisbell::GUID));
	writeByteCount += sizeof(Gravisbell::GUID);

	// �o�b�t�@�֓ǂݍ���
	writeByteCount += neuralNetwork.WriteToBuffer(&lpBuffer[writeByteCount]);
	if(writeByteCount != lpBuffer.size())
		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;

	// �o�b�t�@���t�@�C���֏�������
	{
		// �t�@�C���I�[�v��
		FILE* fp = fopen(filePath.string().c_str(), "wb");
		if(fp == NULL)
			return ErrorCode::ERROR_CODE_COMMON_FILE_NOT_FOUND;

		// ��������
		fwrite(&lpBuffer[0], 1, lpBuffer.size(),fp);

		// �t�@�C���N���[�Y
		fclose(fp);
	}

	return ErrorCode::ERROR_CODE_NONE;
}
/** �j���[�����l�b�g���[�N���o�C�i���t�@�C������ǂݍ��ނ��� */
Gravisbell::ErrorCode ReadNetworkFromBinaryFile(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::ILayerData** ppNeuralNetwork, const wchar_t i_filePath[])
{
	boost::filesystem::path filePath = i_filePath;

	std::vector<BYTE> lpBuffer;
	S32 readByteCount = 0;

	// �t�@�C���̒��g���o�b�t�@�ɃR�s�[����
	{
		// �t�@�C���I�[�v��
		FILE* fp = fopen(filePath.string().c_str(), "rb");
		if(fp == NULL)
			return ErrorCode::ERROR_CODE_COMMON_FILE_NOT_FOUND;

		// �t�@�C���T�C�Y�𒲂ׂăo�b�t�@���쐬����
		fseek(fp, 0, SEEK_END);
		U32 fileSize = ftell(fp);
		lpBuffer.resize(fileSize);

		// �Ǎ�
		fseek(fp, 0, SEEK_SET);
		fread(&lpBuffer[0], 1, fileSize, fp);

		// �t�@�C���N���[�Y
		fclose(fp);
	}

	// ��ʃR�[�h��ǂݍ���
	Gravisbell::GUID typeCode;
	memcpy(&typeCode, &lpBuffer[readByteCount], sizeof(Gravisbell::GUID));
	readByteCount += sizeof(Gravisbell::GUID);

	// DLL���擾
	auto pLayerDLL = layerDLLManager.GetLayerDLLByGUID(typeCode);
	if(pLayerDLL == NULL)
		return ErrorCode::ERROR_CODE_DLL_NOTFOUND;

	// �l�b�g���[�N���쐬
	S32 useBufferCount = 0;
	*ppNeuralNetwork = pLayerDLL->CreateLayerDataFromBuffer(&lpBuffer[readByteCount], (S32)lpBuffer.size()-readByteCount, useBufferCount);
	readByteCount += useBufferCount;

	return ErrorCode::ERROR_CODE_NONE;
}


}	// NeuralNetworkLayer
}	// Utility
}	// Gravisbell
