//=====================================
// ���C���[�ݒ�����쐬����֐��Q
//=====================================
#ifndef __GRAVISBELL_NN_LAYER_CONFIG_H__
#define __GRAVISBELL_NN_LAYER_CONFIG_H__

#ifdef NNLAYERCONFIG_EXPORTS
#define NNLAYERCONFIG_API __declspec(dllexport)
#else
#define NNLAYERCONFIG_API __declspec(dllimport)
#endif

#include"Common/VersionCode.h"

#include"NNLayerInterface/ILayerConfig.h"
#include"ILayerConfigEx.h"
#include"ILayerConfigItemEx_Enum.h"

namespace Gravisbell {
namespace NeuralNetwork {

	/** ��̃��C���[�ݒ�����쐬���� */
	extern "C" NNLAYERCONFIG_API ILayerConfigEx* CreateEmptyLayerConfig(const GUID& layerCode, const VersionCode& versionCode);

	/** �ݒ荀��(����)���쐬���� */
	extern "C" NNLAYERCONFIG_API ILayerConfigItem_Float* CreateLayerCofigItem_Float(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], float minValue, float maxValue, float defaultValue);
	/** �ݒ荀��(����)���쐬���� */
	extern "C" NNLAYERCONFIG_API ILayerConfigItem_Int* CreateLayerCofigItem_Int(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], int minValue, int maxValue, int defaultValue);
	/** �ݒ荀��(������)���쐬���� */
	extern "C" NNLAYERCONFIG_API ILayerConfigItem_String* CreateLayerCofigItem_String(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], const wchar_t szDefaultValue[]);
	/** �ݒ荀��(�_���l)���쐬���� */
	extern "C" NNLAYERCONFIG_API ILayerConfigItem_Bool* CreateLayerCofigItem_Bool(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], bool defaultValue);
	/** �ݒ荀��(�񋓒l)���쐬���� */
	extern "C" NNLAYERCONFIG_API ILayerConfigItemEx_Enum* CreateLayerCofigItem_Enum(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[]);

}	// NeuralNetwork
}	// Gravisbell


#endif // __NN_LAYERCONFIG_H__