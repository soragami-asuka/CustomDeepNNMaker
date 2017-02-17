//======================================
// ���C���[�̐ݒ���ɂ��ċL��
//======================================
#ifndef __LAYER_CONFIG_DATA__H__
#define __LAYER_CONFIG_DATA__H__

#include<string>

#include<boost/filesystem.hpp>

#include<guiddef.h>
#include"NNLayerConfig.h"

namespace CustomDeepNNLibrary
{
	class LayerConfigData
	{
	private:
		GUID guid;	/**< ���ʃR�[�h */
		VersionCode version;	/**< �o�[�W���� */
		std::wstring default_language;	/**< ��{���� */

		std::wstring name;	/**< ���O */
		std::wstring text;	/**< �����e�L�X�g */

		CustomDeepNNLibrary::INNLayerConfigEx* pStructure;	/**< ���C���[�\����`��� */
		CustomDeepNNLibrary::INNLayerConfigEx* pLearn;		/**< �w�K�ݒ��� */

	public:
		/** �R���X�g���N�^ */
		LayerConfigData();
		/** �f�X�g���N�^ */
		~LayerConfigData();

	public:
		/** XML�t�@�C���������ǂݍ���.
			@param	configFilePath	�ǂݍ���XML�t�@�C���̃p�X
			@return	���������ꍇ0���Ԃ�. */
		int ReadFromXMLFile(const boost::filesystem::wpath& configFilePath);
		/** C++����\�[�X�t�@�C���ɕϊ�/�o�͂���.
			.h/.cpp�t�@�C�������������.
			@param	exportDirPath	�o�͐�f�B���N�g���p�X
			@param	fileName		�o�̓t�@�C����.�g���q�͏���.
			@return ���������ꍇ0���Ԃ�. */
		int ConvertToCPPFile(const boost::filesystem::wpath& exportDirPath, const std::wstring& fileName)const;
	};
}


#endif	// __LAYER_CONFIG_DATA__H__
