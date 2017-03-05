#ifdef IODataLayer_EXPORTS
#define IODataLayer_API __declspec(dllexport)
#else
#define IODataLayer_API __declspec(dllimport)
#endif

#include"NNLayerInterface/IIODataLayer.h"


/** ���͐M���f�[�^���C���[���쐬����.GUID�͎������蓖��.CPU����
	@param bufferSize	�o�b�t�@�̃T�C�Y.��float�^�z��̗v�f��.
	@return	���͐M���f�[�^���C���[�̃A�h���X */
extern "C" IODataLayer_API Gravisbell::NeuralNetwork::IIODataLayer* CreateIODataLayerCPU(Gravisbell::IODataStruct ioDataStruct);
/** ���͐M���f�[�^���C���[���쐬����.CPU����
	@param guid			���C���[��GUID.
	@param bufferSize	�o�b�t�@�̃T�C�Y.��float�^�z��̗v�f��.
	@return	���͐M���f�[�^���C���[�̃A�h���X */
extern "C" IODataLayer_API Gravisbell::NeuralNetwork::IIODataLayer* CreateIODataLayerCPUwithGUID(GUID guid, Gravisbell::IODataStruct ioDataStruct);

/** ���͐M���f�[�^���C���[���쐬����.GPU����
	@param guid			���C���[��GUID.
	@param bufferSize	�o�b�t�@�̃T�C�Y.��float�^�z��̗v�f��.
	@return	���͐M���f�[�^���C���[�̃A�h���X */
extern "C" IODataLayer_API Gravisbell::NeuralNetwork::IIODataLayer* CreateIODataLayerGPU(GUID guid, Gravisbell::IODataStruct ioDataStruct);
